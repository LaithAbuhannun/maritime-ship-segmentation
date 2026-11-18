"""Shared held-out metrics; confidence selection never uses test labels."""
from collections import defaultdict
import numpy as np
from .reference.evaluate import (coco_mask_eval, _ground_truth_masks, _mask_api,
    _normalise_rle, _prediction_box, _box_iou_matrix)

def capped_predictions(predictions, max_dets=100):
    groups = defaultdict(list)
    for p in predictions: groups[int(p['image_id'])].append(p)
    return [p for group in groups.values() for p in
            sorted(group, key=lambda p: -p['score'])[:max_dets]]

def scored_matches(truth, predictions, iou=.5, boxes=False):
    """Score-ordered one-to-one matching, calculated once for the threshold sweep."""
    groups = defaultdict(list)
    for p in predictions: groups[int(p['image_id'])].append(p)
    scores, matches = [], []
    for image_id, items in groups.items():
        items.sort(key=lambda p: -p['score'])
        gt = truth.get(image_id, [])
        if boxes:
            matrix = _box_iou_matrix(np.asarray([_prediction_box(p) for p in items]),
                                    np.asarray(gt))
        elif len(gt):
            matrix = _mask_api().iou([_normalise_rle(p['segmentation']) for p in items],
                                     gt, [0]*len(gt))
        else:
            matrix = np.zeros((len(items), 0))
        used = set()
        for index, p in enumerate(items):
            candidates = [j for j in range(len(gt)) if j not in used and matrix[index,j] >= iou]
            if candidates: used.add(max(candidates, key=lambda j: matrix[index,j]))
            scores.append(float(p['score'])); matches.append(bool(candidates))
    return np.asarray(scores), np.asarray(matches), sum(len(v) for v in truth.values())

def at_threshold(scored, threshold, prefix=''):
    scores, matched, count = scored
    selected = scores >= threshold
    tp = int(matched[selected].sum()); n = int(selected.sum())
    precision, recall = tp/n if n else 0., tp/count if count else 0.
    return {prefix+'Precision':100*precision, prefix+'Recall':100*recall,
            prefix+'F1':100*(2*precision*recall/(precision+recall) if precision+recall else 0.),
            'confidence_threshold':float(threshold)}

def choose_threshold(gt_json, predictions):
    truth, _ = _ground_truth_masks(gt_json)
    scored = scored_matches(truth, predictions)
    thresholds = sorted(set(scored[0].tolist()), reverse=True) or [1.]
    return max((at_threshold(scored,t) for t in thresholds),
               key=lambda r:(r['F1'],r['Precision'],r['confidence_threshold']))['confidence_threshold']

def evaluate_masks(gt_json, predictions, threshold, max_dets=100):
    predictions = capped_predictions(predictions, max_dets)
    truth, _ = _ground_truth_masks(gt_json)
    return {**coco_mask_eval(gt_json,predictions,max_dets),
            **at_threshold(scored_matches(truth,predictions),threshold)}

def evaluate_boxes(truth, predictions, threshold, max_dets=100):
    return at_threshold(scored_matches(truth,capped_predictions(predictions,max_dets),boxes=True),
                        threshold, 'box_')
