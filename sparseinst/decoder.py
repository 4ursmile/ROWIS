# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d
from sparseinst.encoder import SPARSE_INST_ENCODER_REGISTRY
from .probhead import ProbObjectnessHead, ProbObjectnessHeadBlock, ProbObjectnessHeadBlockStack
from .dcn import DeformableConv2d
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

SPARSE_INST_DECODER_REGISTRY = Registry("SPARSE_INST_DECODER")
SPARSE_INST_DECODER_REGISTRY.__doc__ = "registry for SparseInst decoder"


def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(
            Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)

class InstanceDeformableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride = 1):
        super(InstanceDeformableConvBlock, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conv1 = nn.Sequential(
            DeformableConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ).to(device=self.device)
        self.conv2 = nn.Sequential(
            DeformableConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels)
        ).to(device=self.device)
        self.downsample = nn.Sequential(
            DeformableConv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels),
        ).to(device=self.device)
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class InstanceDeformableConv(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, kernel_size=3, stride = 1):
        super(InstanceDeformableConv, self).__init__()
        layers = []
        layers.append(InstanceDeformableConvBlock(in_channels, out_channels, kernel_size, stride))
        for _ in range(1, num_blocks):
            layers.append(InstanceDeformableConvBlock(out_channels, out_channels, kernel_size, stride))
        self.layers = nn.Sequential(*layers).to('cuda' if torch.cuda.is_available() else 'cpu')
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, DeformableConv2d):
                m.init_weights()
    def forward(self, x):
        return self.layers(x)
class DeformableIAMSingle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride = 1):
        super(DeformableIAMSingle, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ).to(device=self.device)
    def init_weights(self, value):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                init.constant_(m.bias, value).to(self.device)
                init.normal_(m.weight, std=0.01).to(self.device)
    def forward(self, x):
        out = self.conv(x)
        return out
class DeformableIAMDouble(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride = 1):
        super(DeformableIAMDouble, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ).to(device=self.device)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
        ).to(device=self.device)
        self.downsample = nn.Sequential(
            DeformableConv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels),
        ).to(device=self.device)
        self.relu = nn.ReLU()
    def init_weights(self, value):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                init.constant_(m.bias, value)
                init.normal_(m.weight, std=0.01)
        for m in self.downsample.modules():
            if isinstance(m, DeformableConv2d):
                m.init_weights()
        self.downsample.to(self.device)
        self.conv.to(self.device)
    def forward(self, x, residual):
        residual = x 
        out = self.conv(x)
        out = self.conv2(out)
        residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out
class DeformableIAM(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, kernel_size=3, stride = 1, result_imtermidiate = False):
        super(DeformableIAM, self).__init__()
        self.start_layer = DeformableIAMSingle(in_channels, out_channels, kernel_size, stride)
        middle_layers = []
        for _ in range(0, num_blocks):
            middle_layers.append(DeformableIAMDouble(out_channels, out_channels, kernel_size, stride))
        self.middle_layers = nn.ModuleList(middle_layers)
        self.end_layer = DeformableIAMSingle(out_channels, out_channels, kernel_size, stride)
        self.result_imtermidiate = result_imtermidiate
    def init_weights(self, value):
        self.start_layer.init_weights(value)
        for layer in self.middle_layers:
            layer.init_weights(value)
        self.end_layer.init_weights(value)
    def forward(self, x):
        hs = []
        out= self.start_layer(x)
        hs.append(out)
        for layer in self.middle_layers:
            out = layer(out, x)
            hs.append(out)
        out= self.end_layer(out)
        hs.append(out)
        return torch.stack(hs)
        
class InstanceBranch(nn.Module):

    def __init__(self, cfg, in_channels):
        super().__init__()
        # norm = cfg.MODEL.SPARSE_INST.DECODER.NORM
        dim = cfg.MODEL.SPARSE_INST.DECODER.INST.DIM
        num_convs = cfg.MODEL.SPARSE_INST.DECODER.INST.CONVS
        num_masks = cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS
        kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

        self.inst_convs = InstanceDeformableConv(num_convs, in_channels, dim)
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)

        # outputs
        self.cls_score = nn.Linear(dim, self.num_classes)
        self.mask_kernel = nn.Linear(dim, kernel_dim)
        self.objectness = nn.Linear(dim, 1)
        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob = iam_prob / normalizer[:, :, None]
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        return pred_logits, pred_kernel, pred_scores, iam


class MaskBranch(nn.Module):

    def __init__(self, cfg, in_channels):
        super().__init__()
        dim = cfg.MODEL.SPARSE_INST.DECODER.MASK.DIM
        num_convs = cfg.MODEL.SPARSE_INST.DECODER.MASK.CONVS
        kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        return self.projection(features)


@SPARSE_INST_DECODER_REGISTRY.register()
class BaseIAMDecoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        # add 2 for coordinates
        in_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS + 2

        self.scale_factor = cfg.MODEL.SPARSE_INST.DECODER.SCALE_FACTOR
        self.output_iam = cfg.MODEL.SPARSE_INST.DECODER.OUTPUT_IAM

        self.inst_branch = InstanceBranch(cfg, in_channels)
        self.mask_branch = MaskBranch(cfg, in_channels)

    @torch.no_grad()
    def compute_coordinates_linspace(self, x):
        # linspace is not supported in ONNX
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features):
        coord_features = self.compute_coordinates(features)
        features = torch.cat([coord_features, features], dim=1)
        pred_logits, pred_kernel, pred_scores,  iams = self.inst_branch(features)
        mask_features = self.mask_branch(features)
        iam = iams[-1]
        N = pred_kernel.shape[1]
        # mask_features: BxCxHxW
        B, C, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel, mask_features.view(
            B, C, H * W)).view(B, N, H, W)

        pred_masks = F.interpolate(
            pred_masks, scale_factor=self.scale_factor,
            mode='bilinear', align_corners=False)

        output = {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
            "pred_scores": pred_scores
        }

        if self.output_iam:
            iam = F.interpolate(iam, scale_factor=self.scale_factor,
                                mode='bilinear', align_corners=False)
            output['pred_iam'] = iam

        return output


class GroupInstanceBranch(nn.Module):

    def __init__(self, cfg, in_channels):
        super().__init__()
        dim = cfg.MODEL.SPARSE_INST.DECODER.INST.DIM
        num_convs = cfg.MODEL.SPARSE_INST.DECODER.INST.CONVS
        num_masks = cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS
        kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM
        self.num_groups = cfg.MODEL.SPARSE_INST.DECODER.GROUPS
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

        self.inst_convs = InstanceDeformableConv(num_convs, in_channels, dim)
        # iam prediction, a group conv
        expand_dim = dim * (self.num_groups)

        self.iam_conv = DeformableIAM(self.num_groups, dim, num_masks)
        # self.iam_conv = nn.Conv2d(
        #     dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
        # outputs
        dropout = cfg.MODEL.SPARSE_INST.DECODER.INST.DROPOUT
        self.fc = nn.Sequential(
            nn.Linear(expand_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU()
        )

        self.cls_score = nn.Linear(dim, self.num_classes)
        self.mask_kernel = nn.Linear(dim, kernel_dim)
        self.objectness = nn.Linear(dim, 1)

        self.fc = _get_clones(self.fc, self.num_groups+2)
        self.cls_score = _get_clones(self.cls_score, self.num_groups+2)
        self.mask_kernel = _get_clones(self.mask_kernel, self.num_groups+2)
        self.objectness = _get_clones(self.objectness, self.num_groups+2)

        self.cls_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(kernel_dim*(self.num_groups+2), kernel_dim),
        )
        self.mask_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(kernel_dim*(self.num_groups+2), kernel_dim),
        )
        self.objectness_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(kernel_dim*(self.num_groups+2), kernel_dim),
        )

        self.prior_prob = 0.01
        self._init_weights()
        torch.autograd.set_detect_anomaly(True)

    def _init_weights(self):
        self.inst_convs.init_weights()
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.iam_conv.init_weights(bias_value)
        for cls_score in self.cls_score:
            init.normal_(cls_score.weight, std=0.01)
            init.constant_(cls_score.bias, bias_value)
        for mask_kernel in self.mask_kernel:
            init.normal_(mask_kernel.weight, std=0.01)
            init.constant_(mask_kernel.bias, 0.0)
        for fc in self.fc:
            for module in fc.modules():
                if isinstance(module, nn.Linear):
                    c2_xavier_fill(module)

    def iam_map_to_features(self, iam, features):
        iam_prob = iam.sigmoid()
        B, N = iam_prob.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            iam_prob, features.view(B, C, -1).permute(0, 2, 1))

        inst_features = inst_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        return inst_features
    def forward(self, features):
        features = self.inst_convs(features)
        output_logits = []
        output_masks = []
        output_scores = []
        iams = self.iam_conv(features)
        for i, iam in enumerate(iams):
            inst_features = self.iam_map_to_features(iam, features)
            inst_features = self.fc[i](inst_features)
            pred_logits = self.cls_score[i](inst_features)
            pred_kernel = self.mask_kernel[i](inst_features)
            pred_scores = self.objectness[i](inst_features)
            output_logits.append(pred_logits)
            output_masks.append(pred_kernel)
            output_scores.append(pred_scores)

        output_logits = torch.stack(output_logits, dim=2)
        output_masks = torch.stack(output_masks, dim=2)
        output_scores = torch.stack(output_scores, dim=2)
        output_logits = output_logits.view(output_logits.shape[0], output_logits.shape[1], -1)
        output_masks = output_masks.view(output_masks.shape[0], output_masks.shape[1], -1)
        output_scores = output_scores.view(output_scores.shape[0], output_scores.shape[1], -1)
        output_logits = self.cls_head(output_logits)
        output_masks = self.mask_head(output_masks)
        output_scores = self.objectness_head(output_scores)
        return output_logits, output_masks, output_scores, iams

    # def forward(self, features):
    #     # instance features (x4 convs)
    #     features = self.inst_convs(features)
    #     # predict instance activation maps
    #     iam = self.iam_conv(features)
    #     iam_prob = iam.sigmoid()

    #     B, N = iam_prob.shape[:2]
    #     C = features.size(1)
    #     # BxNxHxW -> BxNx(HW)
    #     iam_prob = iam_prob.view(B, N, -1)
    #     normalizer = iam_prob.sum(-1).clamp(min=1e-6)
    #     iam_prob = iam_prob / normalizer[:, :, None]

    #     # aggregate features: BxCxHxW -> Bx(HW)xC
    #     inst_features = torch.bmm(
    #         iam_prob, features.view(B, C, -1).permute(0, 2, 1))

    #     inst_features = inst_features.reshape(
    #         B, 4, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

    #     inst_features = F.relu_(self.fc(inst_features))
    #     # predict classification & segmentation kernel & objectness
    #     pred_logits = self.cls_score(inst_features)
    #     pred_kernel = self.mask_kernel(inst_features)
    #     pred_scores = self.objectness(inst_features)
    #     return pred_logits, pred_kernel, pred_scores, iam


@SPARSE_INST_DECODER_REGISTRY.register()
class GroupIAMDecoder(BaseIAMDecoder):

    def __init__(self, cfg):
        super().__init__(cfg)
        in_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS + 2
        self.inst_branch = GroupInstanceBranch(cfg, in_channels)


class GroupInstanceSoftBranch(GroupInstanceBranch):

    def __init__(self, cfg, in_channels):
        super().__init__(cfg, in_channels)
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)

        B, N = iam.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            iam_prob, features.view(B, C, -1).permute(0, 2, 1))

        inst_features = inst_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        inst_features = F.relu_(self.fc(inst_features))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        return pred_logits, pred_kernel, pred_scores, iam


@SPARSE_INST_DECODER_REGISTRY.register()
class GroupIAMSoftDecoder(BaseIAMDecoder):

    def __init__(self, cfg):
        super().__init__(cfg)
        in_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS + 2
        self.inst_branch = GroupInstanceSoftBranch(cfg, in_channels)


def build_sparse_inst_decoder(cfg):
    name = cfg.MODEL.SPARSE_INST.DECODER.NAME
    return SPARSE_INST_DECODER_REGISTRY.get(name)(cfg)
