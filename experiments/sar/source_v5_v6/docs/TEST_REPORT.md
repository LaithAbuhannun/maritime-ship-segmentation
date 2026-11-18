# Local verification — 28 August 2026

16 automated tests passed. Local pip check reported no broken requirements.

Coverage:

- Configuration validation and syntax of all three notebook code-cell sets.
- Deterministic split creation and overlap/content-duplicate detection.
- Separate validation/test paths and immutable manifest fingerprints.
- Early-stopping state restoration.
- Required training/validation loss fields and persistent CSV/JSON histories.
- Loss plots written to PNG and PDF.
- ClearML reporting calls contain exactly the local training/validation losses
  (mocked API; no live account upload performed).
- Fixed validation confidence threshold and exact perfect-mask metric fixture.
- Repeated COCO evaluation without mutation of caller-owned predictions.
- Cross-dataset image IDs rejected.
- Validation objective does not update model parameters.
- Real Ship Contour network loss forward/backward on a synthetic CPU fixture.
- Native optimizer/scheduler/scaler/RNG checkpoint fields, stopping and history on resume.
- RF-DETR 1.9.4 configuration and callback contract with a mocked trainer.
- Per-model aggregation and explicit missing LS-SSD rows without copying SSDD scores.
- Multipart training annotations rejected rather than silently split into ships.

Test environment: macOS ARM64, Python 3.12.13, CPU PyTorch 2.13.0,
NumPy 1.26.4, RF-DETR 1.9.4 and Ultralytics 8.4.131.
The tests also passed with NumPy 2.2.6, but that version is excluded by the
Ultralytics macOS dependency marker, so the dependency-checked local environment
uses NumPy 1.26.4. The target modern Linux environment uses NumPy 2.2.6.

These tests do NOT verify the pinned Colab CUDA environments, actual MMDetection
compiled operations, full model training, external SAMSAR integration, live
ClearML authentication/uploads, or bitwise interrupted-versus-uninterrupted GPU runs.
Use the provided two-epoch smoke cells to check these before expensive runs.

The archive contains source, notebooks, configuration and documentation only:
no datasets, trained checkpoints, credentials or previous benchmark results.
