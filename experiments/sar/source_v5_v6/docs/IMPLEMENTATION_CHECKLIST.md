# Readiness checklist

## Implemented

- Persistent disjoint train/validation/test splits and content fingerprints.
- Eight variants with independent run directories.
- Epoch training/validation losses and validation AP.
- Validation-based stopping and best/last checkpoint paths.
- Drive histories, explicit ClearML scalar reporting and history replay.
- Split-specific predictions, dataset/checkpoint identity checks.
- Validation-selected thresholds; separate mask and box result tables.
- Isolated environment setup, smoke-gated notebooks and reports.
- No modifications to old baseline results.

## Still needs Colab verification

- [ ] Each model/dataset smoke run completes with required losses and metrics.
- [ ] ClearML graphs match history.csv and artifacts upload.
- [ ] Interrupt/resume each framework; check epoch, optimizer and patience continue.
- [ ] Inspect validation masks, especially SAMSAR and paper-guided models.
- [ ] Verify external SAMSAR source/version.
- [ ] Verify pinned Linux CUDA environments and selected GPUs.
- [ ] Review scene grouping and split independence.
- [ ] Freeze final settings and disclose framework-specific recipes.

Local tests are not evidence of successful full GPU training. Long-run cells
default to disabled and require a matching successful smoke marker.
