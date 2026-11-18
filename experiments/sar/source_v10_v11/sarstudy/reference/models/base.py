"""Lifecycle contract implemented by every benchmark model adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ModelAdapter(ABC):
    name = "base"
    protocol = "end2end"
    reuses: str | None = None

    def __init__(self, cfg, dataset: str = "ssdd") -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.model: Any | None = None
        self.ckpt: str | None = None
        self.arch_record: dict = {}

    @property
    def run_dir(self) -> str:
        location = Path(self.cfg.runs_dir(self.dataset)) / self.name
        location.mkdir(parents=True, exist_ok=True)
        return str(location)

    def record_architecture(self, actual: dict) -> dict:
        from .architectures import check
        self.arch_record = check(self.name, actual,
                                 strict=self.cfg.strict_architecture)
        return self.arch_record

    @property
    def report_name(self) -> str:
        from .architectures import deviation_suffix
        return self.name + deviation_suffix(self.arch_record)

    @abstractmethod
    def train(self, data: dict) -> None:
        """Fit or otherwise materialise the dataset-specific checkpoint."""

    @abstractmethod
    def load(self, data: dict) -> None:
        """Make the adapter ready for prediction."""

    @abstractmethod
    def predict_coco(self, data: dict) -> list[dict]:
        """Predict the manifest test images and return COCO detections."""

    @abstractmethod
    def predict_one(self, image_path: str):
        """Run the complete inference path for FPS measurement."""

    def params_M(self) -> float | None:
        try:
            parameters = self.model.parameters()
            return round(sum(item.numel() for item in parameters) / 1_000_000, 2)
        except (AttributeError, TypeError):
            return None

    def training_recipe(self) -> dict:
        from .architectures import paper_training_config
        paper = paper_training_config(self.name, self.dataset)
        if self.cfg.use_paper_training_config:
            return {"epochs": paper.get("epochs", self.cfg.epochs),
                    "batch_size": paper.get("batch_size", self.cfg.batch_size),
                    **{key: value for key, value in paper.items()
                       if key not in {"epochs", "batch_size"}}}
        return {"epochs": self.cfg.epochs, "batch_size": self.cfg.batch_size}

    def existing_checkpoint(self) -> str | None:
        candidates = [path for path in Path(self.run_dir).rglob("*")
                      if path.is_file() and path.suffix in {".pt", ".pth", ".ckpt"}]
        return str(max(candidates, key=lambda path: path.stat().st_mtime)) if candidates else None

    def hyperparams(self) -> dict:
        return {"model": self.name, "dataset": self.dataset,
                "protocol": self.protocol, "img_size": self.cfg.img_size,
                "seed": self.cfg.seed,
                "recipe_mode": ("paper" if self.cfg.use_paper_training_config
                                  else "controlled"),
                **self.training_recipe()}


class NotAvailable(ModelAdapter):
    reason = "required dependency is unavailable"

    def _raise(self):
        raise RuntimeError(f"{self.name}: {self.reason}")

    def train(self, data: dict) -> None:
        self._raise()

    def load(self, data: dict) -> None:
        self._raise()

    def predict_coco(self, data: dict) -> list[dict]:
        self._raise()

    def predict_one(self, image_path: str):
        self._raise()

