# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment
from fvcore.nn import sigmoid_focal_loss_jit
import math
import copy
from typing import List, Dict, Tuple

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



class ConfidenceCalibration(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

@SPARSE_INST_CRITERION_REGISTRY.register()
class SparseInstCriterion(nn.Module):
    def __init__(self, cfg, matcher):
        super().__init__()
        self.matcher = matcher
        self.losses = cfg.MODEL.SPARSE_INST.LOSS.ITEMS
        self.weight_dict = self.get_weight_dict(cfg)
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES 
        self.empty_weight = cfg.MODEL.OWIS.EMPTY_WEIGHT
        self.invalid_cls_logits = list(range(cfg.MODEL.OWIS.PREV_INTRODUCED_CLS+ cfg.MODEL.OWIS.CUR_INTRODUCED_CLS, self.num_classes-1))
        self.unknown_class_id = self.num_classes - 1
        self.objectness_threshold = cfg.MODEL.OWIS.OBJECTNESS_THRESHOLD
        self.unknown_loss_weight = cfg.MODEL.OWIS.UNKNOWN_LOSS_WEIGHT
        self.unmatched_weight = cfg.MODEL.OWIS.UNMATCH_WEIGHT

        self.confidence_calibration = ConfidenceCalibration(cfg.MODEL.OWIS.CALIBRATION_TEMPERATURE)

    def get_weight_dict(self, cfg):
        losses = ("loss_ce", "loss_mask", "loss_dice", "loss_objectness", "loss_unknow")
        weight_dict = {}
        ce_weight = cfg.MODEL.SPARSE_INST.LOSS.CLASS_WEIGHT
        mask_weight = cfg.MODEL.SPARSE_INST.LOSS.MASK_PIXEL_WEIGHT
        dice_weight = cfg.MODEL.SPARSE_INST.LOSS.MASK_DICE_WEIGHT
        objectness_weight = cfg.MODEL.SPARSE_INST.LOSS.OBJECTNESS_WEIGHT
        unknown_weight = cfg.MODEL.OWIS.UNKNOWN_LOSS_WEIGHT
        weight_dict = dict(
            zip(losses, (ce_weight, mask_weight, dice_weight, objectness_weight, unknown_weight)))
        return weight_dict

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, num_instances, input_shape=None):
        assert "pred_logits" in outputs
        temp_src_logits = outputs['pred_logits'].clone()
        temp_src_logits[:,:, self.invalid_cls_logits] = -10e10
        src_logits = temp_src_logits

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
       
        target_classes = torch.full(src_logits.shape[:2], self.num_classes - 1, dtype=torch.int64, device=src_logits.device)
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

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_instances, alpha=0.25,
                                     num_classes=self.num_classes, empty_weight=self.empty_weight) * calibrated_logits.shape[1]

        # Additional loss for unknown class
        unknown_logits = calibrated_logits[:, :, self.unknown_class_id]
        unknown_targets = (target_classes == self.unknown_class_id).float()
        loss_unknown = F.binary_cross_entropy_with_logits(unknown_logits, unknown_targets, reduction='mean')

        losses = {
            'loss_ce': loss_ce,
            'loss_unknown': loss_unknown 
        }
        return losses

    def loss_masks_with_iou_objectness(self, outputs, targets, indices, num_instances, input_shape):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        assert "pred_masks" in outputs
        assert "pred_scores" in outputs
        src_iou_scores = outputs["pred_scores"]
        src_masks = outputs["pred_masks"]
        with torch.no_grad():
            target_masks, _ = nested_masks_from_list(
                [t["masks"].tensor for t in targets], input_shape).decompose()
        num_masks = [len(t["masks"]) for t in targets]
        target_masks = target_masks.to(src_masks)
        if len(target_masks) == 0:
            losses = {
                "loss_dice": src_masks.sum() * 0.0,
                "loss_mask": src_masks.sum() * 0.0,
                "loss_objectness": src_iou_scores.sum() * 0.0
            }
            return losses

        src_masks = src_masks[src_idx]
        target_masks = F.interpolate(
            target_masks[:, None], size=src_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

        src_masks = src_masks.flatten(1)
        mix_tgt_idx = torch.zeros_like(tgt_idx[1])
        cum_sum = 0
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]

        target_masks = target_masks[mix_tgt_idx].flatten(1)

        with torch.no_grad():
            ious = compute_mask_iou(src_masks, target_masks)

        tgt_iou_scores = torch.zeros(src_iou_scores.shape, dtype=ious.dtype).to(src_iou_scores)
        tgt_iou_scores = tgt_iou_scores.to(ious)
        tgt_iou_scores[src_idx] = ious.view(-1, 1)

        # Handle unmatched predictions
        unmatched_mask = torch.ones(src_iou_scores.shape[:2], dtype=torch.bool, device=src_iou_scores.device)
        unmatched_mask[src_idx[0], src_idx[1]] = False

        # Adaptive thresholding for unknown objects
        objectness_scores = src_iou_scores.sigmoid().flatten(1)
        unknown_mask = (objectness_scores > self.objectness_threshold) & unmatched_mask

        # Assign average IoU score to unknown predictions
        avg_iou = ious.min()
        tgt_iou_scores[unknown_mask] = avg_iou



        # Other unmatched predictions are assigned the empty_weight
        tgt_iou_scores[unmatched_mask & ~unknown_mask] = self.unmatched_weight

   


        tgt_iou_scores = tgt_iou_scores.flatten(0)
        src_iou_scores = src_iou_scores.flatten(0)

        losses = {
            "loss_objectness": F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean'),
            "loss_dice": dice_loss(src_masks, target_masks) / num_instances,
            "loss_mask": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean'),
        }
        return losses

    # def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
    #     assert "pred_obj" in outputs
    #     idx = self._get_src_permutation_idx(indices)
    #     pred_obj = outputs["pred_obj"][idx]
    #     return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks_with_iou_objectness,
            # "obj_likelihood": self.loss_obj_likelihood,
        }
        if loss == "loss_objectness":
            return {}
        assert loss in loss_map
        return loss_map[loss](outputs, targets, indices, num_instances, **kwargs)

    def forward(self, outputs, targets, input_shape):
        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets, input_shape)
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = torch.as_tensor(
            [num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_instances)
        num_instances = torch.clamp(
            num_instances / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices,
                                        num_instances, input_shape=input_shape))

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return losses



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
            with torch.autocast("cuda", enabled=False):
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