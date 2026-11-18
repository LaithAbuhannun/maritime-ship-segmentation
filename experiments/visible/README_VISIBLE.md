# Visible Ship Study

Upload `Visible_Ship_Study_Package.zip` and `Visible_Ship_Dataset.zip` into the main MyDrive directory. Open the separate notebooks in Colab and run setup cells in order.

1. `01_Visible_Modern_Models.ipynb`: YOLO26 n/s/m, RF-DETR Seg Nano, SAMSAR, Ship Contour.
2. `02_Visible_MMDetection_Models.ipynb`: SwinInsSeg, SOLOv2-GALA. Python 3.11; A100/T4 GPU.
3. `03_Visible_Results.ipynb`: result tables, saved loss curves, same-image qualitative comparisons.

## Shared protocol

One frozen seed-42 subset is used for all eight variants: 1,000 training, 250 validation and 500 test images. The original Roboflow `valid` source groups provide the test pool; the source training pool supplies the training/validation groups. Images linked by original filename or identical decoded RGB pixels are grouped; identical-pixel copies are removed. All known source groups and pixel hashes are disjoint across the final splits. This does not establish independence of unidentified photographs from the same scene or near-duplicates without matching source names.

All ship subtypes are collapsed to one `ship` category, matching the SAR task. Raw annotations are not modified. The prepared bundle stores original filenames, polygons, hashes and the selection audit. Common training controls are 500 maximum epochs, patience 20, batch size 1, workers 0, AMP off and validation mask AP50:95 for checkpoint selection/early stopping. A confidence threshold is selected on validation, then frozen for testing. Test images never enter training or patience. Normal input size is 512; RF-DETR requires 504 to satisfy its architecture's multiple-of-56 constraint.

Optimizers retain model-specific requirements: SOLOv2 uses SGD at 0.01 (momentum 0.9, weight decay 0.0001); SwinInsSeg uses AdamW at 0.00002; other models use AdamW at 0.0001. These are controlled settings, not a claim of optimal tuning or guaranteed nonzero AP. Architecture adapters are the same paper-guided implementations used in the SAR study; they have not been independently validated as exact paper reproductions.

SAMSAR uses ground-truth bounding-box prompts and is labelled `ground_truth_box_prompted`. Its masks have the same segmentation evaluator, but its input protocol differs from autonomous detection models. It requires your existing `SAMSAR_Implementation` folder in MyDrive and the notebook's SAM checkpoint prerequisite cell.

## Saved outputs

`runs/visible/<model>/visible_v1_seed42/` holds per-epoch numeric `history.csv` and `history.json`, training/validation loss figures (PNG/PDF), validation AP, best and last checkpoints, validation/test predictions, validation-selected threshold, final metrics, configuration, provenance and completion markers.

Final segmentation metrics include AP50, AP50:95, AP75, AP by object size, precision, recall, F1, FPS and parameter count. Metrics that have no applicable ground-truth instances may be unavailable. With one class, class-averaged AP equals that class's AP. The common COCO evaluator is mask based.

ClearML uses the new project `Visible Ship Segmentation Study`. Each model has its own task; numeric training/validation loss is reported against the epoch value. History, figures, metrics and checkpoints are uploaded. Resumed executions currently create a new attempt task and replay the full durable history to that task, so the latest task displays the complete curve. Colab must authenticate successfully; no silent local-only fallback is enabled.

## Resume

After a new session, rerun mount/extraction, environment setup, credentials and the run helper. Rerun the relevant full-training model cell. Completed runs skip training, and interrupted runs load their saved state. Keep configuration and package versions fixed during a run; identity checks reject changed-code resumes. This visible study has its own folder and output project. Existing SAR results are not copied into its metrics.

## Verification

The dataset selection and conversions are checked locally, including source/pixel/hash disjointness, polygon validity, one-class COCO conversion, fixed IDs and identical manifests across model configurations. Full CUDA training and live ClearML authentication must be checked in Colab through the supplied two-epoch smoke cells.
