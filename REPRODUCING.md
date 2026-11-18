# Reproduce and inspect

## 1. Inspect the evidence without a GPU

```bash
git clone https://github.com/LaithAbuhannun/maritime-ship-segmentation.git
cd maritime-ship-segmentation
python3 scripts/inspect_results.py --dataset ssdd
python3 scripts/validate_portfolio.py
python3 -m http.server 8000 --directory docs
```

Open http://localhost:8000 to explore the report and results. The inspection and validation scripts use the Python standard library. CSV/JSON metrics, all 24 retained histories, native ClearML curve grids and qualitative examples are available without installing training frameworks.

## 2. Prepare data

Obtain data from the original sources in [CREDITS.md](CREDITS.md). The SAR copies were supplied by Mahdi Rahimi; do not assume a current public release is byte-identical. The visible export was `visible-ship-dataset` version 1. Configure the dataset paths before preparing data. Use the recorded grouping/hash audits and seed 42. Training/validation/test counts in the report are SSDD 733/183/232, HRSID 1000/250/500, visible 1000/250/500. These are selected subsets, not all available images.

## 3. Choose the historical source and configuration

Use [experiments/README.md](experiments/README.md) and [selected_runs.json](results/selected_runs.json). For example, RF-DETR on SSDD uses the `source_v5_v6` SAR source and its recorded `study_v5_leakage_safe` configuration. Archived default configs are not a substitute for the per-run record.

Each source tree contains a `pyproject.toml`, environment requirements, `scripts/setup_environment.py`, and `sarstudy.runner`. Follow the notebook setup cells with the chosen tree uploaded to its expected Drive location. A typical command, **inside that tree's configured environment**, is:

```bash
python -u -m sarstudy.runner --config /path/to/recorded-config.json --model rfdetr_seg --dataset ssdd --smoke
```

The final run omits `--smoke`. First validate the small smoke run, losses, masks and environment. The research used a maximum of 500 epochs, patience 20 and validation-selected checkpoints and confidence thresholds. These controls do not make all model recipes identical. Test data must not select checkpoints, thresholds or patience.

## 4. GPU/environment constraints

Modern YOLO/RF-DETR environments and the pinned MMDetection environment are separate. The recorded MMDetection stack requires Python 3.11 and a compatible CUDA GPU such as A100/T4; its historical Torch/MMCV binaries do not support Blackwell. Do not blindly upgrade the legacy stack. Check the retained `environment.json`/requirements where available and the experiment environment files. SAMSAR requires externally obtained source and the SAM ViT-B checkpoint.

## 5. Credentials and expected outputs

Authenticate ClearML locally using its normal setup process or notebook secrets; never commit keys. Per-run output includes `config.json`, split/data manifests, `history.csv/json`, checkpoints, `threshold.json`, validation/test predictions, metrics, environment records and completion markers. Full datasets and weights are intentionally not in this repository. The lightweight scripts above were checked locally; GPU retraining was not repeated for this portfolio.
