"""Small compatibility helpers shared by MMDetection adapters."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from ..evaluate import detection

MMDET_HINT = ("Install compatible MMDetection dependencies with openmim: "
              "mmengine, mmcv>=2.0, and mmdet>=3.0.")


def mmdet_base_config(relative_path: str) -> str:
    """Resolve a model-zoo config from an installed MMDetection package."""
    try:
        import mmdet
    except ImportError as error:
        raise RuntimeError(MMDET_HINT) from error
    package = Path(mmdet.__file__).resolve().parent
    candidates = (package / ".mim" / "configs" / relative_path,
                  package / "configs" / relative_path,
                  package.parent / "configs" / relative_path)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"MMDetection config {relative_path!r} is not present in {package}; "
        "install the package with its model-zoo configs via openmim")


def require_training_api():
    try:
        from mmengine.config import Config as MMConfig
        from mmengine.runner import Runner
    except ImportError as error:
        raise RuntimeError(MMDET_HINT) from error
    return MMConfig, Runner


def newest_checkpoint(folder: str) -> str:
    preferred = list(Path(folder).glob("best*.pth"))
    candidates = preferred or list(Path(folder).glob("*.pth"))
    if not candidates:
        candidates = list(Path(folder).rglob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No MMDetection checkpoint found in {folder}")
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def restore_runner(config_path: str, patch_model, checkpoint: str):
    MMConfig, Runner = require_training_api()
    config = MMConfig.fromfile(config_path)
    runner = Runner.from_cfg(config)
    patch_model(runner.model)
    try:
        from mmengine.runner.checkpoint import load_checkpoint
        load_checkpoint(runner.model, checkpoint, map_location="cpu", strict=True)
    except ImportError as error:
        raise RuntimeError(MMDET_HINT) from error
    try:
        import torch
        runner.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    except ImportError:
        pass
    runner.model.cfg = config
    runner.model.eval()
    return runner.model


def coco_predictions(model, manifest: dict, score_threshold: float) -> list[dict]:
    try:
        from mmdet.apis import inference_detector
    except ImportError as error:
        raise RuntimeError(MMDET_HINT) from error
    payload = json.loads(Path(manifest["coco_test"]).read_text(encoding="utf-8"))
    output = []
    for info in payload.get("images", []):
        result = inference_detector(
            model, str(Path(manifest["test_image_dir"]) / info["file_name"]))
        instances = result.pred_instances
        masks = getattr(instances, "masks", None)
        if masks is None:
            continue
        masks = masks.detach().cpu().numpy().astype(np.uint8)
        scores = instances.scores.detach().cpu().numpy()
        for mask, score in zip(masks, scores):
            if score >= score_threshold and mask.any():
                output.append(detection(info["id"], mask, float(score)))
    return output


def infer_one(model, image_path: str):
    try:
        from mmdet.apis import inference_detector
    except ImportError as error:
        raise RuntimeError(MMDET_HINT) from error
    return inference_detector(model, image_path)

