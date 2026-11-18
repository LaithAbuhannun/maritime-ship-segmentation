"""Shared mask and box evaluation for every benchmark adapter."""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

METRIC_COLUMNS = ["AP50", "AP50_95", "Precision", "Recall", "F1",
                  "FPS", "AP_small"]


def _mask_api():
    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise ImportError("Mask evaluation requires pycocotools") from error
    return mask_utils


def mask_to_rle(mask: np.ndarray) -> dict:
    binary = np.asfortranarray(np.asarray(mask, dtype=np.uint8))
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {binary.shape}")
    encoded = _mask_api().encode(binary)
    if isinstance(encoded["counts"], bytes):
        encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def detection(image_id: int, mask: np.ndarray, score: float) -> dict:
    if not 0 <= score <= 1:
        raise ValueError("Detection confidence must be between zero and one")
    return {"image_id": int(image_id), "category_id": 1,
            "segmentation": mask_to_rle(mask), "score": float(score)}


def coco_mask_eval(gt_json: str, detections: list[dict],
                   max_dets: int = 100) -> dict:
    """Return COCO mask AP as percentages, preserving unavailable bands as None."""
    labels = ["AP50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
    if not detections:
        result = {label: 0.0 for label in labels}
        result.update({"mAP": 0.0, "AP": 0.0, "num_detections": 0,
                       "ap_curve": {}})
        return result
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise ImportError("COCO evaluation requires pycocotools") from error
    ground_truth = COCO(gt_json)
    # COCO.loadRes adds bbox/area/id in place. Keep caller-owned mask predictions
    # immutable and use mask area (not box area) for mask AP size categories.
    import copy
    clean = [{k: copy.deepcopy(v) for k,v in item.items()
              if k not in {'bbox','area','id','iscrowd'}} for item in detections]
    predictions = ground_truth.loadRes(clean)
    evaluator = COCOeval(ground_truth, predictions, iouType="segm")
    evaluator.params.maxDets = [1, 10, int(max_dets)]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    # COCO's summary layout is only defined for the conventional maxDets=100.
    # For another cap, compute the headline mean directly from precision.
    stats = list(map(float, evaluator.stats[:6]))
    result = {name: (None if value < 0 else value * 100)
              for name, value in zip(labels, stats)}
    precision = evaluator.eval.get("precision")
    curve = {}
    if precision is not None:
        def mean_precision(area_index: int = 0,
                           threshold_index: int | None = None):
            values = (precision[:, :, :, area_index, -1] if threshold_index is None
                      else precision[threshold_index, :, :, area_index, -1])
            valid = values[values >= 0]
            return float(valid.mean() * 100) if valid.size else None

        for threshold_index, threshold in enumerate(evaluator.params.iouThrs):
            values = precision[threshold_index, :, :, 0, -1]
            valid = values[values >= 0]
            curve[f"{threshold:.2f}"] = (float(valid.mean() * 100)
                                          if valid.size else None)
        thresholds = np.asarray(evaluator.params.iouThrs)
        index_50 = int(np.argmin(np.abs(thresholds - .50)))
        index_75 = int(np.argmin(np.abs(thresholds - .75)))
        # Compute directly from the accumulated precision tensor. This works
        # for any configured max_dets, unlike COCOeval.summarize(), whose six
        # AP slots are hard-coded to maxDets=100.
        result.update({"AP50_95": mean_precision(0),
                       "AP50": mean_precision(0, index_50),
                       "AP75": mean_precision(0, index_75),
                       "AP_small": mean_precision(1),
                       "AP_medium": mean_precision(2),
                       "AP_large": mean_precision(3)})
    result["AP"] = result["AP50_95"]
    result["mAP"] = result["AP50_95"]
    result["num_detections"] = len(detections)
    result["ap_curve"] = curve
    return result


def _normalise_rle(segmentation: dict) -> dict:
    value = dict(segmentation)
    if isinstance(value.get("counts"), str):
        value["counts"] = value["counts"].encode("ascii")
    return value


def _ground_truth_masks(gt_json: str) -> tuple[dict[int, list[dict]], int]:
    payload = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    image_shapes = {int(item["id"]): (int(item["height"]), int(item["width"]))
                    for item in payload.get("images", [])}
    grouped: dict[int, list[dict]] = defaultdict(list)
    mask_utils = _mask_api()
    for annotation in payload.get("annotations", []):
        image_id = int(annotation["image_id"])
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, list):
            height, width = image_shapes[image_id]
            rles = mask_utils.frPyObjects(segmentation, height, width)
            grouped[image_id].append(mask_utils.merge(rles))
        elif isinstance(segmentation, dict):
            grouped[image_id].append(_normalise_rle(segmentation))
    return grouped, sum(map(len, grouped.values()))


def _greedy_mask_matches(truth: list[dict], predicted: list[dict],
                         iou_threshold: float) -> int:
    if not truth or not predicted:
        return 0
    mask_utils = _mask_api()
    matrix = mask_utils.iou([_normalise_rle(item["segmentation"])
                             for item in predicted], truth,
                            [0] * len(truth))
    candidates = [(-float(matrix[row, col]), row, col)
                  for row in range(matrix.shape[0])
                  for col in range(matrix.shape[1])
                  if matrix[row, col] >= iou_threshold]
    used_predictions, used_truth, matches = set(), set(), 0
    for _, row, col in sorted(candidates):
        if row not in used_predictions and col not in used_truth:
            used_predictions.add(row)
            used_truth.add(col)
            matches += 1
    return matches


def _best_operating_point(scored_counts: Iterable[tuple[float, int, int, int]],
                          prefix: str = "") -> dict:
    best = (0.0, 0.0, 0.0, 0.0, 0.0)
    for threshold, true_positive, predicted_count, truth_count in scored_counts:
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / truth_count if truth_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        candidate = (f1, precision, recall, threshold, true_positive)
        if candidate > best:
            best = candidate
    f1, precision, recall, threshold, _ = best
    return {f"{prefix}Precision": precision * 100,
            f"{prefix}Recall": recall * 100,
            f"{prefix}F1": f1 * 100,
            "conf_at_best_F1": threshold}


def prf1(gt_json: str, detections: list[dict],
         iou_thr: float = .5) -> dict:
    ground_truth, truth_count = _ground_truth_masks(gt_json)
    if truth_count == 0:
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0,
                "conf_at_best_F1": 0.0}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for item in detections:
        grouped[int(item["image_id"])].append(item)
    thresholds = sorted({float(item.get("score", 1.0)) for item in detections},
                        reverse=True) or [1.0]
    counts = []
    for threshold in thresholds:
        true_positive = predicted_count = 0
        for image_id in set(ground_truth) | set(grouped):
            selected = [item for item in grouped.get(image_id, [])
                        if float(item.get("score", 1.0)) >= threshold]
            predicted_count += len(selected)
            true_positive += _greedy_mask_matches(
                ground_truth.get(image_id, []), selected, iou_thr)
        counts.append((threshold, true_positive, predicted_count, truth_count))
    return _best_operating_point(counts)


def _box_iou_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64).reshape(-1, 4)
    second = np.asarray(second, dtype=np.float64).reshape(-1, 4)
    if first.size == 0 or second.size == 0:
        return np.zeros((len(first), len(second)), dtype=np.float64)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    first_area = np.clip(first[:, 2] - first[:, 0], 0, None) * np.clip(
        first[:, 3] - first[:, 1], 0, None)
    second_area = np.clip(second[:, 2] - second[:, 0], 0, None) * np.clip(
        second[:, 3] - second[:, 1], 0, None)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _prediction_box(item: dict) -> list[float] | None:
    if len(item.get("bbox", [])) == 4:
        x, y, width, height = map(float, item["bbox"])
        return [x, y, x + width, y + height]
    segmentation = item.get("segmentation")
    if isinstance(segmentation, dict):
        x, y, width, height = map(float, _mask_api().toBbox(
            _normalise_rle(segmentation)))
        return [x, y, x + width, y + height]
    return None


def box_prf1(gt_boxes_by_image: dict[int, np.ndarray], detections: list[dict],
             iou_thr: float = .5) -> dict:
    truth_count = sum(len(value) for value in gt_boxes_by_image.values())
    grouped: dict[int, list[tuple[list[float], float]]] = defaultdict(list)
    for item in detections:
        box = _prediction_box(item)
        if box is not None:
            grouped[int(item["image_id"])].append((box, float(item.get("score", 1))))
    thresholds = sorted({score for values in grouped.values() for _, score in values},
                        reverse=True) or [1.0]
    counts = []
    for threshold in thresholds:
        matches = predicted_count = 0
        for image_id in set(gt_boxes_by_image) | set(grouped):
            truth = np.asarray(gt_boxes_by_image.get(image_id, []), dtype=float).reshape(-1, 4)
            selected = [box for box, score in grouped.get(image_id, []) if score >= threshold]
            predicted_count += len(selected)
            matrix = _box_iou_matrix(np.asarray(selected).reshape(-1, 4), truth)
            candidates = [(-float(matrix[row, col]), row, col)
                          for row in range(matrix.shape[0]) for col in range(matrix.shape[1])
                          if matrix[row, col] >= iou_thr]
            used_rows, used_cols = set(), set()
            for _, row, col in sorted(candidates):
                if row not in used_rows and col not in used_cols:
                    used_rows.add(row); used_cols.add(col); matches += 1
        counts.append((threshold, matches, predicted_count, truth_count))
    return _best_operating_point(counts, prefix="box_")


def benchmark_fps(predict_one: Callable[[str], object], image_paths: list[str],
                  max_images: int = 100, warmup: int = 10) -> float:
    chosen = image_paths[:max_images]
    if not chosen:
        return 0.0
    for image_path in chosen[:min(warmup, len(chosen))]:
        predict_one(image_path)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    started = time.perf_counter()
    for image_path in chosen:
        predict_one(image_path)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    elapsed = time.perf_counter() - started
    return len(chosen) / elapsed if elapsed > 0 else math.inf


def results_table(named: dict[str, dict]):
    import pandas as pd
    rows = {}
    for model_name, metrics in named.items():
        rows[model_name] = {"protocol": metrics.get("protocol"),
                            "fidelity": (metrics.get("fidelity") or
                                         metrics.get("arch_fidelity")),
                            **{key: metrics.get(key) for key in METRIC_COLUMNS},
                            "conf@F1": metrics.get("conf_at_best_F1"),
                            "params_M": metrics.get("params_M")}
    columns = ["protocol", "fidelity", *METRIC_COLUMNS, "conf@F1", "params_M"]
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=columns)


def generalization_table(named: dict[str, dict]):
    import pandas as pd
    columns = ["box_Precision", "box_Recall", "box_F1", "conf_at_best_F1",
               "FPS", "source_dataset"]
    return pd.DataFrame.from_dict(named, orient="index").reindex(columns=columns)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_results(named: dict[str, dict], out_dir: str,
                 basename: str = "results") -> dict[str, str]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"{basename}.json"
    csv_path = output / f"{basename}.csv"
    json_path.write_text(json.dumps(_json_safe(named), indent=2), encoding="utf-8")
    try:
        table = (generalization_table(named) if any("box_F1" in row for row in named.values())
                 else results_table(named))
        table.to_csv(csv_path)
    except ImportError:
        csv_path.write_text("pandas is required for CSV output\n", encoding="utf-8")
    return {f"{basename}_metrics_json": str(json_path),
            f"{basename}_csv": str(csv_path)}
