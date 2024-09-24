import numpy as np
import pycocotools.mask as mask_util
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import create_small_table
from detectron2.utils import comm  # Add this import
from tabulate import tabulate
import itertools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    has_mask = instances.has("pred_masks")
    if has_mask:
        # Convert masks to RLE format
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        # Generate bounding boxes from masks
        boxes = [mask_util.toBbox(rle).tolist() for rle in rles]
    else:
        return []

    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "score": scores[k],
            "segmentation": rles[k],
            "bbox": boxes[k],
        }
        results.append(result)
    return results

class COCOMaskEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, tasks, distributed, output_dir=None):
        super().__init__(dataset_name, tasks, distributed, output_dir)
        self._mask_predictions = []
        self._bbox_predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
                
                # Separate mask and bbox predictions
                for p in prediction["instances"]:
                    self._mask_predictions.append({
                        "image_id": p["image_id"],
                        "category_id": p["category_id"],
                        "segmentation": p["segmentation"],
                        "score": p["score"],
                    })
                    self._bbox_predictions.append({
                        "image_id": p["image_id"],
                        "category_id": p["category_id"],
                        "bbox": p["bbox"],
                        "score": p["score"],
                    })

            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            mask_predictions = comm.gather(self._mask_predictions, dst=0)
            bbox_predictions = comm.gather(self._bbox_predictions, dst=0)
            predictions = comm.gather(self._predictions, dst=0)
            mask_predictions = list(itertools.chain(*mask_predictions))
            bbox_predictions = list(itertools.chain(*bbox_predictions))
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            mask_predictions = self._mask_predictions
            bbox_predictions = self._bbox_predictions
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOMaskEvaluator] Did not receive valid predictions.")
            return {}

        # Evaluate masks
        mask_results = self._eval_predictions(mask_predictions, "segm")
        
        # Evaluate bounding boxes derived from masks
        bbox_results = self._eval_predictions(bbox_predictions, "bbox")
        
        # Combine results
        results = {
            "segm": mask_results,
            "bbox": bbox_results
        }
        
        return results

    def _eval_predictions(self, coco_results, iou_type):
        coco_dt = self._coco_api.loadRes(coco_results)
        coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return self._derive_coco_results(coco_eval, iou_type, self._metadata.get("thing_classes"))

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR", "AR50", "AR75", "ARs", "ARm", "ARl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR", "AR50", "AR75", "ARs", "ARm", "ARl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {} {}: \n".format(iou_type, "masks" if iou_type == "segm" else "boxes") + 
            create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and are shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        
        # Compute per-category AP and AR
        precisions = coco_eval.eval["precision"]
        recalls = coco_eval.eval["recall"]
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            
            recall = recalls[:, idx, 0, -1]
            recall = recall[recall > -1]
            ar = np.mean(recall) if recall.size else float("nan")
            
            results_per_category.append(
                ("{}".format(name), float(ap * 100), float(ar * 100))
            )

        # Tabulate it
        N_COLS = min(6, len(results_per_category) * 3)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP", "AR"] * (N_COLS // 3),
            numalign="left",
        )
        self._logger.info("Per-category {} AP and AR for {} {}: \n".format(iou_type, iou_type, "masks" if iou_type == "segm" else "boxes") + table)

        results.update({"AP-" + name: ap for name, ap, _ in results_per_category})
        results.update({"AR-" + name: ar for name, _, ar in results_per_category})
        return results