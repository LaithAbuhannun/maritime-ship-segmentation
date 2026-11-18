# Problems addressed from the earlier project

- Missing training/validation losses: explicit required epoch schema, local CSV/JSON,
  plotted curves, and ClearML reporting independent of framework auto-detection.
- Test used as validation: validation now comes from the training pool only.
- Test-optimized F1: threshold selection moved to validation.
- Shared dataset CSV overwritten by the next model: per-model metrics are the
  source of truth; reports only read and combine them.
- SSDD evaluation resumed with LS-SSD image IDs: predictions now have separate
  split filenames and split/checkpoint fingerprints; unexpected IDs are rejected.
- MMDetection model missing cfg: the constructed model receives its inference config.
- COCO info key missing: generated annotations include info/licenses.
- COCO evaluation mutating predictions: evaluation receives an isolated copy.
- Cached directories existing but incomplete: manifests and image hashes are checked.
- Lost work after disconnect: best and full resume state are stored in Drive.
- Python/package conflicts: isolated environments, direct compatible MMCV wheel install,
  and no openmim dependency chain or forced reinstall of the Colab kernel.
- GPU OOM: conservative explicit batch size; automatic YOLO batch changes are rejected
  rather than silently changing the experiment. GPU smoke checks remain mandatory.
- All results apparently present but from the wrong dataset: provenance is checked;
  missing rows remain missing and box/mask tables stay separate.

Research-specific safeguards: SAMSAR is explicitly ground-truth-box-prompted;
framework-specific recipes are disclosed; FPS carries GPU identity; multipart/RLE
training annotations are rejected rather than silently split into multiple ships.
