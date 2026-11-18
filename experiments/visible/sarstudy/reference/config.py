"""Configuration and protocol constants for the ship benchmark."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_DATA_ROOT = Path(os.getenv("SAR_DATA_ROOT", "/content/data"))
DEFAULT_WORK_ROOT = Path(os.getenv("SAR_WORK_ROOT", "/content/work"))
PROMPTED_MODELS = frozenset({"samsar"})
INFERENCE_VARIANTS = {
    "yolo26_seg_sahi": "yolo26_seg",
    "rfdetr_seg_sahi": "rfdetr_seg",
}


def _split_sizes() -> dict[str, dict[str, int | None]]:
    return {"ssdd": {"train": 928, "test": 232},
            "hrsid": {"train": 3642, "test": 1962}}


def _model_keys() -> list[str]:
    return ["yolo26_seg", "rfdetr_seg", "samsar", "swininsseg",
            "solov2_gala", "ship_contour"]


def _subset_sizes() -> dict[str, dict[str, int | None]]:
    """Resource-bounded final protocol agreed for the honours benchmark."""
    return {"ssdd": {"train": None, "test": None},
            "hrsid": {"train": 1000, "test": 500},
            "visible": {"train": None, "test": None}}


@dataclass(slots=True)
class Config:
    """Complete, serialisable description of one benchmark run."""
    ssdd_dir: str = str(DEFAULT_DATA_ROOT / "SSDD_coco")
    hrsid_dir: str = str(DEFAULT_DATA_ROOT / "HRSID")
    ls_ssdd_dir: str = str(DEFAULT_DATA_ROOT / "LS-SSD")
    visible_dir: str = str(DEFAULT_DATA_ROOT / "visible")
    work_dir: str = str(DEFAULT_WORK_ROOT)
    prepared_data_dir: str | None = None
    train_datasets: list[str] = field(default_factory=lambda: ["ssdd", "hrsid"])
    generalization_dataset: str | None = "ls_ssdd"
    generalization_source_dataset: str = "ssdd"
    generalization_only_with_ships: bool = True
    generalization_max_images: int | None = 500
    generalization_samples_to_save: int = 12
    expected_splits: dict[str, dict[str, int | None]] = field(default_factory=_split_sizes)
    epochs: int = 50
    batch_size: int = 8
    seed: int = 42
    workers: int = 8
    img_size: int = 512
    samsar_img_size: int = 256
    sahi_slice: int = 256
    sahi_overlap: float = .20
    sahi_postprocess: str = "NMM"
    sahi_match_metric: str = "IOS"
    sahi_conf: float = .10
    score_threshold: float = .05
    max_dets: int = 100
    prf1_iou: float = .50
    fps_images: int = 100
    fps_warmup: int = 10
    clearml_project: str = "Honours - SAR Ship Segmentation"
    clearml_run_name: str = "benchmark"
    use_clearml: bool = True
    strict_architecture: bool = True
    rfdetr_variant: str = "nano"
    swin_variant: str = "tiny"
    use_paper_training_config: bool = True
    dataset_subset_sizes: dict[str, dict[str, int | None]] = field(
        default_factory=_subset_sizes)
    reuse_prepared_data: bool = True
    resume_existing: bool = True
    fail_on_zero_predictions: bool = True
    samsar_repo: str = "/content/SAMSAR_Implementation"
    sam_checkpoint: str = "/content/sam_vit_b_01ec64.pth"
    models: list[str] = field(default_factory=_model_keys)
    DOMAINS = {"ssdd": "sar", "hrsid": "sar", "ls_ssdd": "sar",
               "visible": "visible"}

    def __post_init__(self) -> None:
        if not 0 <= self.sahi_overlap < 1:
            raise ValueError("sahi_overlap must be in [0, 1)")
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        unknown = set(self.train_datasets) - {"ssdd", "hrsid", "visible"}
        if unknown:
            raise ValueError(f"Unsupported training datasets: {sorted(unknown)}")

    def subset_size(self, dataset: str, split: str) -> int | None:
        return self.dataset_subset_sizes.get(dataset, {}).get(split)

    def domain_of(self, dataset: str) -> str:
        return self.DOMAINS.get(dataset, "unknown")

    def dataset_dir(self, dataset: str) -> str:
        paths = {"ssdd": self.ssdd_dir, "hrsid": self.hrsid_dir,
                 "ls_ssdd": self.ls_ssdd_dir, "visible": self.visible_dir}
        try:
            return paths[dataset]
        except KeyError as error:
            raise KeyError(f"Unknown dataset: {dataset}") from error

    def expected(self, dataset: str) -> dict[str, int | None]:
        return self.expected_splits.get(dataset, {"train": None, "test": None})

    def coco_dir(self, dataset: str) -> str:
        root = Path(self.prepared_data_dir or self.work_dir)
        return str(root / "coco" / dataset)

    def yolo_dir(self, dataset: str) -> str:
        root = Path(self.prepared_data_dir or self.work_dir)
        return str(root / "yolo" / dataset)

    def runs_dir(self, dataset: str) -> str:
        return str(Path(self.work_dir) / "runs" / dataset)

    def results_dir(self) -> str:
        return str(Path(self.work_dir) / "results")

    def ensure_dirs(self) -> None:
        Path(self.results_dir()).mkdir(parents=True, exist_ok=True)
        datasets = set(self.train_datasets)
        if self.generalization_dataset:
            datasets.add(self.generalization_dataset)
        for dataset in datasets:
            for location in (self.coco_dir(dataset), self.yolo_dir(dataset),
                             self.runs_dir(dataset)):
                Path(location).mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

