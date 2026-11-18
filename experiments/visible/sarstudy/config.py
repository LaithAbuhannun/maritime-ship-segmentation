from dataclasses import dataclass, asdict, field
from pathlib import Path
import re

MODELS = ("yolo26n_seg", "yolo26s_seg", "yolo26m_seg", "rfdetr_seg",
          "samsar", "swininsseg", "solov2_gala", "ship_contour")

@dataclass
class StudyConfig:
    project_root: str = "/content/drive/MyDrive/Visible_Ship_Study"
    data_root: str = "/content/drive/MyDrive/Visible_Ship_Study/data"
    cache_root: str = "/content/visible_ship_cache"
    experiment: str = "visible_v1_seed42"
    seed: int = 42
    validation_fraction: float = .2
    # 1,250 source-training images -> 1,000 train + 250 validation at 20%.
    hrsid_pool: int | None = 1250
    hrsid_test: int | None = 500
    generalization_images: int = 500
    max_epochs: int = 500
    patience: int = 20
    min_delta: float = .1  # percentage points of validation mask AP50:95
    batch_size: int = 2
    workers: int = 4
    img_size: int = 512
    rfdetr_resolution: int = 504
    samsar_img_size: int = 256
    amp: bool = True
    smoke: bool = False
    learning_rate: float = 1e-4
    prediction_floor: float = .001
    max_dets: int = 100
    fps_images: int = 100
    fps_warmup: int = 10
    clearml: bool = True
    clearml_project: str = "SAR Ship Segmentation Study"
    samsar_repo: str = "/content/drive/MyDrive/SAMSAR_Implementation"
    sam_checkpoint: str = "/content/sam_vit_b_01ec64.pth"
    datasets: list[str] = field(default_factory=lambda: ["visible"])
    models: list[str] = field(default_factory=lambda: list(MODELS))

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.experiment):
            raise ValueError("Use letters, numbers, '_' or '-' for experiment")
        if not 0 < self.validation_fraction < .5:
            raise ValueError("validation_fraction must be in (0, .5)")
        if min(self.max_epochs, self.patience, self.batch_size) < 1 or self.min_delta < 0:
            raise ValueError("Invalid epoch, patience, batch size or min_delta")
        if self.workers < 0 or not 0 <= self.prediction_floor < 1:
            raise ValueError("Invalid workers or prediction floor")
        if set(self.models) - set(MODELS) or set(self.datasets) - {"ssdd", "hrsid", "visible"}:
            raise ValueError("Unknown model or dataset")
        if self.rfdetr_resolution % 56:
            raise ValueError("RF-DETR resolution must be divisible by 56")
        if self.smoke and self.max_epochs > 3:
            raise ValueError('Smoke runs are limited to three epochs')

    def as_dict(self):
        return asdict(self)

    def run_dir(self, dataset, model):
        if dataset not in {"ssdd", "hrsid", "visible"} or model not in MODELS:
            raise ValueError("Invalid run identity")
        return Path(self.project_root) / "runs" / dataset / model / self.experiment
