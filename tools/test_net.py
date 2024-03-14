import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.engine import default_argument_parser, default_setup
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, print_csv_format
from detectron2.data.datasets import register_coco_instances

sys.path.append(".")
from sparseinst import build_sparse_inst_encoder, build_sparse_inst_decoder, add_sparse_inst_config
from sparseinst import COCOMaskEvaluator


device = torch.device('cuda:0')
dtype = torch.float32

__all__ = ["SparseInst"]

pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).to(device).view(3, 1, 1)
pixel_std = torch.Tensor([58.395, 57.120, 57.375]).to(device).view(3, 1, 1)


@torch.jit.script
def normalizer(x, mean, std): return (x - mean) / std


def synchronize():
    torch.cuda.synchronize()


def process_batched_inputs(batched_inputs):
    images = [x["image"].to(device) for x in batched_inputs]
    images = [normalizer(x, pixel_mean, pixel_std) for x in images]
    images = ImageList.from_tensors(images, 32)
    ori_size = (batched_inputs[0]["height"], batched_inputs[0]["width"])
    return images.tensor, images.image_sizes[0], ori_size


@torch.jit.script
def rescoring_mask(scores, mask_pred, masks):
    mask_pred_ = mask_pred.float()
    return scores * ((masks * mask_pred_).sum([1, 2]) / (mask_pred_.sum([1, 2]) + 1e-6))


class SparseInst(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        # backbone
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        output_shape = self.backbone.output_shape()

        self.encoder = build_sparse_inst_encoder(cfg, output_shape)
        self.decoder = build_sparse_inst_decoder(cfg)

        self.to(self.device)

        # inference
        self.cls_threshold = cfg.MODEL.SPARSE_INST.CLS_THRESHOLD
        self.mask_threshold = cfg.MODEL.SPARSE_INST.MASK_THRESHOLD
        self.max_detections = cfg.MODEL.SPARSE_INST.MAX_DETECTIONS
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

        self.invalid_cls_logits = list(range(cfg.MODEL.OWIS.PREV_INTRODUCED_CLS+ cfg.MODEL.OWIS.CUR_INTRODUCED_CLS, self.num_classes-1))
        self.temperature = cfg.MODEL.OWIS.TEMPERATURE
        self.pred_per_image = cfg.MODEL.OWIS.PRED_PER_IMAGE

    def forward(self, image, resized_size, ori_size):
        max_size = image.shape[2:]
        features = self.backbone(image)
        features = self.encoder(features)
        output = self.decoder(features)
        result = self.inference_single(
            output, resized_size, max_size, ori_size)
        return result

    def inference_single(self, outputs, img_shape, pad_shape, ori_shape):
        """
        inference for only one sample
        Args:
            scores (tensor): [NxC]
            masks (tensor): [NxHxW]
            img_shape (list): (h1, w1), image after resized
            pad_shape (list): (h2, w2), padded resized image
            ori_shape (list): (h3, w3), original shape h3*w3 < h1*w1 < h2*w2
        """
        result = Instances(ori_shape)
        # scoring
        pred_logits = outputs["pred_logits"][0]
        pred_scores = outputs["pred_scores"][0]
        pred_masks = outputs["pred_masks"][0].sigmoid()
        print(pred_logits.shape, pred_scores.shape, pred_masks.shape)
        pred_logits[:, self.invalid_cls_logits] = -10e10

        obj_prob = torch.exp(-self.temperature*pred_scores).unsqueeze(-1)
        prob = obj_prob * torch.sigmoid(pred_logits)
        print(prob.shape)
        print(prob.view(pred_logits[0].shape[0], -1))
        print(prob.view(pred_logits[0].shape[0], -1).shape)
        top_k_prob, top_k_idx = torch.topk(prob.view(pred_logits.shape[0], -1), 10, dim=1)
        print(top_k_prob, top_k_idx)
        scores = top_k_prob
        masks = top_k_idx // pred_logits.shape[1]
        labels = top_k_idx % pred_logits.shape[1]
        print(masks, labels)
        # obtain scores
        scores, labels = pred_logits.max(dim=-1)
        # remove by thresholding
        keep = scores > self.cls_threshold
        scores = torch.sqrt(scores[keep] * pred_scores[keep])
        labels = labels[keep]
        pred_masks = pred_masks[keep]

        if scores.size(0) == 0:
            return None
        scores = rescoring_mask(scores, pred_masks > 0.45, pred_masks)
        h, w = img_shape
        # resize masks
        pred_masks = F.interpolate(pred_masks.unsqueeze(1), size=pad_shape,
                                   mode="bilinear", align_corners=False)[:, :, :h, :w]
        pred_masks = F.interpolate(pred_masks, size=ori_shape, mode='bilinear',
                                   align_corners=False).squeeze(1)
        mask_pred = pred_masks > self.mask_threshold

        mask_pred = BitMasks(mask_pred)
        result.pred_masks = mask_pred
        result.scores = scores
        result.pred_classes = labels
        return result


def test_sparseinst_speed(cfg, fp16=False):
    device = torch.device('cuda:0')

    model = SparseInst(cfg)
    model.eval()
    model.to(device)
    print(model)
    size = (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False)

    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = False

    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

    evaluator = COCOMaskEvaluator(
        cfg.DATASETS.TEST[0], ("segm",), False, output_folder)
    evaluator.reset()
    model.to(device)
    model.eval()
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    durations = []

    with autocast(enabled=fp16):
        with torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                images, resized_size, ori_size = process_batched_inputs(inputs)
                synchronize()
                start_time = time.perf_counter()
                output = model(images, resized_size, ori_size)
                synchronize()
                end = time.perf_counter() - start_time

                durations.append(end)
                if idx % 1000 == 0:
                    print("process: [{}/{}] fps: {:.3f}".format(idx,
                                                                len(data_loader), 1/np.mean(durations[100:])))
                evaluator.process(inputs, [{"instances": output}])
    # evaluate
    results = evaluator.evaluate()
    print_csv_format(results)

    latency = np.mean(durations[100:])
    fps = 1 / latency
    print("speed: {:.4f}s FPS: {:.2f}".format(latency, fps))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if not args.eval_only:
        cfg.SOLVER.IMS_PER_BATCH = 32
        cfg.SOLVER.MAX_ITER = 27000
        cfg.BASE_LR = 5e-5/4 # default batch size is 64
    cfg.DATASETS.TRAIN = ('minitest_train',)
    cfg.DATASETS.TEST = ("minitest_valid",)   
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == '__main__':


    for d in ["valid", "train"]:
      register_coco_instances("minitest_" + d, {}, f"datasets/minitest_{d}/_annotations.coco.json", "datasets/minitest_"+d)
    args = default_argument_parser()
    args.add_argument("--fp16", action="store_true",
                      help="support fp16 for inference")
    args = args.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    test_sparseinst_speed(cfg, fp16=args.fp16)
