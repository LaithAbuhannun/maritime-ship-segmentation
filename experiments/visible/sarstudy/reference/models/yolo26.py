"""Ultralytics YOLO26 instance segmentation and its SAHI inference variant."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from ..evaluate import detection
from .base import ModelAdapter

PAPER_WEIGHTS = "yolo26n-seg.pt"
SUBSTITUTES = ("yolo11n-seg.pt", "yolov8n-seg.pt")


def _resolve_weights(preferred: str | None = None,
                     strict: bool = True) -> str:
    from ultralytics import YOLO
    requested = preferred or PAPER_WEIGHTS
    try:
        YOLO(requested)
        return requested
    except Exception as original_error:
        if strict:
            raise RuntimeError(
                f"YOLO26 checkpoint {requested!r} is unavailable; refusing "
                "to substitute another architecture") from original_error
    for candidate in SUBSTITUTES:
        try:
            YOLO(candidate)
            return candidate
        except Exception:
            continue
    raise RuntimeError("No usable Ultralytics segmentation checkpoint")


def _polygons_to_mask(polygons, height: int, width: int) -> np.ndarray:
    from PIL import Image, ImageDraw
    canvas = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        points = np.asarray(polygon)
        if points.ndim == 2 and len(points) >= 3:
            draw.polygon([tuple(value) for value in np.rint(points).astype(int)],
                         fill=1)
    return np.asarray(canvas, dtype=np.uint8)


class YOLO26Seg(ModelAdapter):
    name = "yolo26_seg"

    def __init__(self, cfg, dataset: str = "ssdd",
                 weights: str | None = None) -> None:
        super().__init__(cfg, dataset)
        self.pretrained = weights
        self.weights_used: str | None = None

    def train(self, data: dict) -> None:
        from ultralytics import YOLO
        self.weights_used = _resolve_weights(self.pretrained,
                                             self.cfg.strict_architecture)
        self.record_architecture({"weights": self.weights_used,
                                  "backbone": "YOLO26", "detector": "YOLO26",
                                  "mask_head": "YOLO26 segmentation head",
                                  "variant": "n"})
        self.model = YOLO(self.weights_used)
        recipe = self.training_recipe()
        self.model.train(data=data["yolo_yaml"], epochs=recipe["epochs"],
                         imgsz=self.cfg.img_size, batch=recipe["batch_size"],
                         seed=self.cfg.seed, workers=self.cfg.workers,
                         project=self.run_dir, name="train", exist_ok=True,
                         deterministic=True, val=True, plots=False)
        checkpoint = Path(self.run_dir) / "train" / "weights" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Ultralytics did not create {checkpoint}")
        self.ckpt = str(checkpoint)

    def load(self, data: dict) -> None:
        from ultralytics import YOLO
        if self.ckpt is None:
            trained = Path(self.run_dir) / "train" / "weights" / "best.pt"
            self.ckpt = str(trained) if trained.is_file() else _resolve_weights(
                self.pretrained, self.cfg.strict_architecture)
        self.model = YOLO(self.ckpt)
        if not self.arch_record:
            declared = self.weights_used or self.pretrained or PAPER_WEIGHTS
            self.record_architecture({"weights": declared, "backbone": "YOLO26",
                                      "detector": "YOLO26",
                                      "mask_head": "YOLO26 segmentation head",
                                      "variant": "n"})

    def _predict_path(self, image_path: str):
        if self.model is None:
            raise RuntimeError("YOLO model is not loaded")
        return self.model.predict(image_path, imgsz=self.cfg.img_size,
                                  conf=self.cfg.score_threshold,
                                  max_det=self.cfg.max_dets, verbose=False)[0]

    def predict_coco(self, data: dict) -> list[dict]:
        payload = json.loads(Path(data["coco_test"]).read_text(encoding="utf-8"))
        output = []
        for image in payload.get("images", []):
            prediction = self._predict_path(
                str(Path(data["test_image_dir"]) / image["file_name"]))
            masks = getattr(prediction, "masks", None)
            if masks is None:
                continue
            scores = (prediction.boxes.conf.detach().cpu().numpy()
                      if getattr(prediction, "boxes", None) is not None else [])
            for index, polygon in enumerate(masks.xy):
                binary = _polygons_to_mask([polygon], image["height"], image["width"])
                if binary.any():
                    score = float(scores[index]) if index < len(scores) else 1.
                    output.append(detection(image["id"], binary, score))
        return output

    def predict_one(self, image_path: str):
        return self._predict_path(image_path)


class YOLO26SegSAHI(YOLO26Seg):
    name = "yolo26_seg_sahi"
    reuses = "yolo26_seg"

    def train(self, data: dict) -> None:
        raise RuntimeError("SAHI variants reuse their base model checkpoint")

    def load(self, data: dict) -> None:
        from .sahi_wrap import build_sahi_model
        if not self.ckpt:
            base = Path(self.cfg.runs_dir(self.dataset)) / "yolo26_seg" / "train" / "weights" / "best.pt"
            if not base.is_file():
                raise FileNotFoundError(f"Base YOLO26 checkpoint not found: {base}")
            self.ckpt = str(base)
        self.model = build_sahi_model(self.ckpt, self.cfg, "ultralytics")
        self.record_architecture({"backbone": "YOLO26", "detector": "YOLO26",
                                  "mask_head": "YOLO26 segmentation head",
                                  "variant": "n", "inference": "SAHI"})

    def predict_coco(self, data: dict) -> list[dict]:
        from .sahi_wrap import sliced_predict_coco
        return sliced_predict_coco(self.model, data, self.cfg, "yolo26+sahi")

    def predict_one(self, image_path: str):
        from .sahi_wrap import sliced_predict_one
        return sliced_predict_one(self.model, image_path, self.cfg)

    def hyperparams(self) -> dict:
        return {**super().hyperparams(), "reuses_checkpoint_of": self.reuses,
                "sahi_slice": self.cfg.sahi_slice,
                "sahi_overlap": self.cfg.sahi_overlap,
                "sahi_postprocess": self.cfg.sahi_postprocess}

