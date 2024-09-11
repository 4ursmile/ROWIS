# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment
from fvcore.nn import sigmoid_focal_loss_jit
import math
import copy


from detectron2.utils.registry import Registry

from .utils import nested_masks_from_list, is_dist_avail_and_initialized, get_world_size

SPARSE_INST_MATCHER_REGISTRY = Registry("SPARSE_INST_MATCHER")
SPARSE_INST_MATCHER_REGISTRY.__doc__ = "Matcher for SparseInst"
SPARSE_INST_CRITERION_REGISTRY = Registry("SPARSE_INST_CRITERION")
SPARSE_INST_CRITERION_REGISTRY.__doc__ = "Criterion for SparseInst"
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, num_classes: int = 81, empty_weight: float = 0.1):
    prob = inputs.sigmoid()
    W = torch.ones(num_classes, dtype=prob.dtype, layout=prob.layout, device=prob.device)
    W[-1] = empty_weight
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=W)

    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.4).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_score(inputs, targets):
    inputs = inputs.sigmoid()
    numerator = 2 * torch.matmul(inputs, targets.t())
    denominator = (
        inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


def dice_loss(inputs, targets, reduction='sum'):
    inputs = inputs.sigmoid()
    assert inputs.shape == targets.shape
    numerator = 2 * (inputs * targets).sum(1)
    denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
    loss = 1 - (numerator) / (denominator + 1e-4)
    if reduction == 'none':
        return loss
    return loss.sum()


@SPARSE_INST_CRITERION_REGISTRY.register()
class SparseInstCriterion(nn.Module):
    def __init__(self, cfg, matcher):
        super().__init__()
        self.matcher = matcher
        self.losses = cfg.MODEL.SPARSE_INST.LOSS.ITEMS
        self.weight_dict = self.get_weight_dict(cfg)
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES 
        self.empty_weight = cfg.MODEL.OWIS.EMPTY_WEIGHT
        self.invalid_cls_logits = list(range(cfg.MODEL.OWIS.PREV_INTRODUCED_CLS + cfg.MODEL.OWIS.CUR_INTRODUCED_CLS, self.num_classes-1))
        self.unknown_class_id = self.num_classes - 1
        self.objectness_threshold = cfg.MODEL.OWIS.OBJECTNESS_THRESHOLD
        self.unknown_loss_weight = cfg.MODEL.OWIS.UNKNOWN_LOSS_WEIGHT
        self.memory_bank = PrototypeMemoryBank(cfg.MODEL.OWIS.MEMORY_BANK_SIZE, cfg.MODEL.SPARSE_INST.DECODER.HIDDEN_DIM, self.num_classes)
        self.confidence_calibration = ConfidenceCalibration(cfg.MODEL.OWIS.CALIBRATION_TEMPERATURE)
        self.unknown_to_known_threshold = cfg.MODEL.OWIS.UNKNOWN_TO_KNOWN_THRESHOLD
        self.contrastive_loss_weight = cfg.MODEL.OWIS.CONTRASTIVE_LOSS_WEIGHT

    def get_weight_dict(self, cfg):
        losses = ("loss_ce", "loss_mask", "loss_dice", "loss_objectness", "loss_unknown", "loss_contrastive")
        weight_dict = {}
        ce_weight = cfg.MODEL.SPARSE_INST.LOSS.CLASS_WEIGHT
        mask_weight = cfg.MODEL.SPARSE_INST.LOSS.MASK_PIXEL_WEIGHT
        dice_weight = cfg.MODEL.SPARSE_INST.LOSS.MASK_DICE_WEIGHT
        objectness_weight = cfg.MODEL.SPARSE_INST.LOSS.OBJECTNESS_WEIGHT
        unknown_weight = cfg.MODEL.OWIS.UNKNOWN_LOSS_WEIGHT
        contrastive_weight = cfg.MODEL.OWIS.CONTRASTIVE_LOSS_WEIGHT

        weight_dict = dict(zip(losses, (ce_weight, mask_weight, dice_weight, objectness_weight, unknown_weight, contrastive_weight)))
        return weight_dict

    def loss_labels(self, outputs, targets, indices, num_instances, input_shape=None):
        src_logits = outputs['pred_logits'].clone()
        src_logits[:,:, self.invalid_cls_logits] = -10e10

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
       
        target_classes = torch.full(src_logits.shape[:2], self.unknown_class_id, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # Handle unmatched predictions
        objectness_scores = outputs['pred_scores'].sigmoid().flatten(1)
        matched_mask = torch.zeros(src_logits.shape[:2], dtype=torch.bool, device=src_logits.device)
        matched_mask[idx] = True
        unmatched_mask = ~matched_mask

        # Adaptive thresholding for unknown objects
        unknown_mask = (objectness_scores > self.objectness_threshold) & unmatched_mask
        target_classes[unknown_mask] = self.unknown_class_id

        # Apply confidence calibration
        calibrated_logits = self.confidence_calibration(src_logits)

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        loss_ce = sigmoid_focal_loss(calibrated_logits, target_classes_onehot, num_instances, alpha=0.25,
                                     gamma=2.0, num_classes=self.num_classes) * calibrated_logits.shape[1]

        # Additional loss for unknown class
        unknown_logits = calibrated_logits[:, :, self.unknown_class_id]
        unknown_targets = (target_classes == self.unknown_class_id).float()
        loss_unknown = F.binary_cross_entropy_with_logits(unknown_logits, unknown_targets, reduction='mean')

        losses = {
            'loss_ce': loss_ce,
            'loss_unknown': loss_unknown * self.unknown_loss_weight
        }
        return losses

    def loss_masks_with_iou_objectness(self, outputs, targets, indices, num_instances, input_shape):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_iou_scores = outputs["pred_scores"]
        src_masks = outputs["pred_masks"]
        src_embeddings = outputs["pred_embeddings"]

        with torch.no_grad():
            target_masks, _ = nested_masks_from_list([t["masks"].tensor for t in targets], input_shape).decompose()
        
        num_masks = [len(t["masks"]) for t in targets]
        target_masks = target_masks.to(src_masks)
        
        if len(target_masks) == 0:
            return {
                "loss_dice": src_masks.sum() * 0.0,
                "loss_mask": src_masks.sum() * 0.0,
                "loss_objectness": src_iou_scores.sum() * 0.0
            }

        src_masks = src_masks[src_idx]
        target_masks = F.interpolate(target_masks[:, None], size=src_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)

        with torch.no_grad():
            ious = compute_mask_iou(src_masks, target_masks)

        tgt_iou_scores = torch.zeros(src_iou_scores.shape, dtype=ious.dtype, device=src_iou_scores.device)
        tgt_iou_scores[src_idx] = ious.view(-1, 1)

        # Handle unmatched predictions
        unmatched_mask = torch.ones(src_iou_scores.shape[:2], dtype=torch.bool, device=src_iou_scores.device)
        unmatched_mask[src_idx[0], src_idx[1]] = False

        # Adaptive thresholding for unknown objects
        objectness_scores = src_iou_scores.sigmoid().flatten(1)
        unknown_mask = (objectness_scores > self.objectness_threshold) & unmatched_mask

        # Assign average IoU score to unknown predictions
        avg_iou = ious.mean()
        tgt_iou_scores[unknown_mask] = avg_iou

        # Other unmatched predictions are assigned the empty_weight
        tgt_iou_scores[unmatched_mask & ~unknown_mask] = self.empty_weight
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # Update memory bank and get contrastive loss
        contrastive_loss = self.memory_bank.update_and_contrast(src_embeddings[src_idx], ious, target_classes_o)

        tgt_iou_scores = tgt_iou_scores.flatten(0)
        src_iou_scores = src_iou_scores.flatten(0)

        losses = {
            "loss_objectness": F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean'),
            "loss_dice": dice_loss(src_masks, target_masks) / num_instances,
            "loss_mask": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean'),
            "loss_contrastive": contrastive_loss * self.contrastive_loss_weight
        }
        return losses

    def forward(self, outputs, targets, input_shape):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets, input_shape)
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_instances)
        num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_instances, input_shape=input_shape))

        # Check for unknown to known transitions
        self.check_unknown_to_known_transition()

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return losses

    def check_unknown_to_known_transition(self):
        unknown_prototypes = self.memory_bank.get_unknown_prototypes()
        for class_id, prototypes in unknown_prototypes.items():
            if len(prototypes) > self.unknown_to_known_threshold:
                self.move_unknown_to_known(class_id)

    def move_unknown_to_known(self, class_id):
        # Logic to move an unknown class to a known class
        # This might involve updating self.invalid_cls_logits, 
        # adjusting the model's classification layer, etc.
        if class_id in self.invalid_cls_logits:
            self.invalid_cls_logits.remove(class_id)
        print(f"Moving unknown class {class_id} to known category")

class PrototypeMemoryBank:
    def __init__(self, size: int, feature_dim: int, num_classes: int):
        self.size = size
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.features = torch.zeros(size, feature_dim)
        self.labels = torch.zeros(size, dtype=torch.long)
        self.ious = torch.zeros(size)
        self.index = 0
        self.prototypes = {i: [] for i in range(num_classes)}

    def update_and_contrast(self, new_features: torch.Tensor, new_ious: torch.Tensor, new_labels: torch.Tensor) -> torch.Tensor:
        num_new = new_features.size(0)
        if self.index + num_new > self.size:
            self.index = 0
        
        end_index = min(self.index + num_new, self.size)
        self.features[self.index:end_index] = new_features[:end_index-self.index]
        self.ious[self.index:end_index] = new_ious[:end_index-self.index]
        self.labels[self.index:end_index] = new_labels[:end_index-self.index]
        self.index = end_index

        # Update prototypes
        for i in range(self.num_classes):
            class_features = self.features[self.labels == i]
            if len(class_features) > 0:
                self.prototypes[i] = class_features.mean(0, keepdim=True)

        # Compute contrastive loss
        similarities = torch.mm(new_features, self.features.t())
        positive_similarities = similarities[torch.arange(num_new), self.labels == new_labels.unsqueeze(1)]
        negative_similarities = similarities[torch.arange(num_new), self.labels != new_labels.unsqueeze(1)]
        
        contrastive_loss = -torch.log(torch.exp(positive_similarities) / 
                                      (torch.exp(positive_similarities) + torch.exp(negative_similarities).sum(1)))
        return contrastive_loss.mean()

    def get_unknown_prototypes(self) -> Dict[int, List[torch.Tensor]]:
        return {i: self.prototypes[i] for i in range(self.num_classes) if i not in self.invalid_cls_logits}

class ConfidenceCalibration(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature



@SPARSE_INST_MATCHER_REGISTRY.register()
class SparseInstMatcherV1(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.alpha = cfg.MODEL.SPARSE_INST.MATCHER.ALPHA
        self.beta = cfg.MODEL.SPARSE_INST.MATCHER.BETA
        self.mask_score = dice_score

    @torch.no_grad()
    def forward(self, outputs, targets, input_shape):
        B, N, H, W = outputs["pred_masks"].shape
        pred_masks = outputs['pred_masks']
        pred_logits = outputs['pred_logits'].sigmoid()

        indices = []

        for i in range(B):
            tgt_ids = targets[i]["labels"]
            # no annotations
            if tgt_ids.shape[0] == 0:
                indices.append((torch.as_tensor([]),
                                torch.as_tensor([])))
                continue

            tgt_masks = targets[i]['masks'].tensor.to(pred_masks)
            pred_logit = pred_logits[i]
            out_masks = pred_masks[i]

            # upsampling:
            # (1) padding/
            # (2) upsampling to 1x input size (input_shape)
            # (3) downsampling to 0.25x input size (output mask size)
            ori_h, ori_w = tgt_masks.size(1), tgt_masks.size(2)
            tgt_masks_ = torch.zeros(
                (1, tgt_masks.size(0), input_shape[0], input_shape[1])).to(pred_masks)
            tgt_masks_[0, :, :ori_h, :ori_w] = tgt_masks
            tgt_masks = F.interpolate(
                tgt_masks_, size=out_masks.shape[-2:], mode='bilinear', align_corners=False)[0]

            # compute dice score and classification score
            tgt_masks = tgt_masks.flatten(1)
            out_masks = out_masks.flatten(1)

            mask_score = self.mask_score(out_masks, tgt_masks)
            # Nx(Number of gts)
            matching_prob = pred_logit[:, tgt_ids]
            C = (mask_score ** self.alpha) * (matching_prob ** self.beta)
            # hungarian matching
            inds = linear_sum_assignment(C.cpu(), maximize=True)
            indices.append(inds)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


@SPARSE_INST_MATCHER_REGISTRY.register()
class SparseInstMatcher(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.alpha = cfg.MODEL.SPARSE_INST.MATCHER.ALPHA
        self.beta = cfg.MODEL.SPARSE_INST.MATCHER.BETA
        self.mask_score = dice_score

    def forward(self, outputs, targets, input_shape):
        with torch.no_grad():
            B, N, H, W = outputs["pred_masks"].shape
            pred_masks = outputs['pred_masks']
            pred_logits = outputs['pred_logits'].sigmoid()

            tgt_ids = torch.cat([v["labels"] for v in targets])

            if tgt_ids.shape[0] == 0:
                return [(torch.as_tensor([]).to(pred_logits), torch.as_tensor([]).to(pred_logits))] * B
            tgt_masks, _ = nested_masks_from_list(
                [t["masks"].tensor for t in targets], input_shape).decompose()
            device = pred_masks.device
            tgt_masks = tgt_masks.to(pred_masks)

            tgt_masks = F.interpolate(
                tgt_masks[:, None], size=pred_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

            pred_masks = pred_masks.view(B * N, -1)
            tgt_masks = tgt_masks.flatten(1)
            with autocast(enabled=False):
                pred_masks = pred_masks.float()
                tgt_masks = tgt_masks.float()
                pred_logits = pred_logits.float()
                mask_score = self.mask_score(pred_masks, tgt_masks)
                # Nx(Number of gts)
                matching_prob = pred_logits.view(B * N, -1)[:, tgt_ids]
                C = (mask_score ** self.alpha) * (matching_prob ** self.beta)

            C = C.view(B, N, -1).cpu()
            # hungarian matching
            sizes = [len(v["masks"]) for v in targets]
            indices = [linear_sum_assignment(c[i], maximize=True)
                       for i, c in enumerate(C.split(sizes, -1))]
            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(
                j, dtype=torch.int64)) for i, j in indices]
            return indices


def build_sparse_inst_matcher(cfg):
    name = cfg.MODEL.SPARSE_INST.MATCHER.NAME
    return SPARSE_INST_MATCHER_REGISTRY.get(name)(cfg)


def build_sparse_inst_criterion(cfg):
    matcher = build_sparse_inst_matcher(cfg)
    name = cfg.MODEL.SPARSE_INST.LOSS.NAME
    return SPARSE_INST_CRITERION_REGISTRY.get(name)(cfg, matcher)