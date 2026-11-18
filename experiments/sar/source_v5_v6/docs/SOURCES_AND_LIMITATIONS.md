# Sources and implementation limits

API references checked:

- https://rfdetr.roboflow.com/latest/learn/train/customization/
- https://rfdetr.roboflow.com/latest/learn/train/training-parameters/
- https://docs.ultralytics.com/usage/callbacks/
- https://mmdetection.readthedocs.io/en/latest/get_started.html
- https://pytorch.org/get-started/previous-versions/
- https://github.com/facebookresearch/segment-anything

RF-DETR 1.9.4 and Ultralytics 8.4.131 wheel source was inspected for callback
ordering, metrics and checkpoint behavior. Those versions are pinned.

sarstudy/reference contains isolated helpers from the prior benchmark. Old
training entrypoints are disabled by the new adapters. The new drivers own
splits, training, validation, logging, checkpointing and evaluation. Some unused
legacy helper classes remain; SAHI is absent from the new model registry.

SwinInsSeg, SOLOv2-GALA and Ship Contour are paper-guided reimplementations,
not independently verified exact reproductions. SAMSAR uses external source
supplied by the user and ground-truth box prompts.

A shared budget/stopping policy does not mean identical optimization recipes.
YOLO uses native augmentation/warmup and EMA validation; RF-DETR native
augmentation/layer-wise settings without EMA; native adapters use AdamW/cosine
and resize-only inputs. Sizes are 512, RF-DETR 504 (patch-grid requirement), and
SAMSAR 256. Disclose these differences in the thesis.

Validation has no gradient/optimizer updates. RF-DETR mask loss may sample
points, so validation-loss variation remains. Native reductions are
sample-weighted; framework losses keep native reductions.

Resume restores saved state, but bitwise equivalence after interruption is not
claimed with multi-worker augmentation and mixed precision. Only load trusted
study checkpoints, not untrusted pickle files.

Local tests are CPU checks. Actual CUDA, compiled MMCV, GPU memory, framework
integration and signed-in ClearML upload require the provided smoke checks.
