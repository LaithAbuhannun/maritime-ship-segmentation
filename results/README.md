# Selected final evidence

`selected_test_metrics.json` and `segmentation_metrics.csv` contain 22 completed, selected test-result rows: seven per SAR dataset and eight for visible light. SAMSAR rows use ground-truth box prompts and are not autonomous detection results. Failed SAR SOLOv2 runs are excluded from the main ranking but their zero-AP histories remain in `histories/` (24 histories total).

`ls_ssdd_box_metrics.json` contains external **box** precision/recall/F1 and FPS for six SSDD-trained autonomous models on 500 ship-containing LS-SSD images. It is not a mask AP benchmark and cannot establish the false-alarm rate on empty scenes.

Values in the presentation are read directly from these retained result files. Floating-point precision is retained in JSON/CSV; figures and tables round for display. Visible Ship Contour's AP is approximately 0.0000307%, so a two-decimal table displays 0.00.

Source: final selected research run exports retained for the November 2026 honours report. SHA-256 checkpoint/split identifiers and training selection metadata are preserved where available. Private Drive locations and ClearML task identifiers are removed from public metric JSON; these identifiers are not required to inspect the reported measurements. Reported FPS used different GPUs and input settings across some SAR models. One seed was used; rankings are descriptive, without confidence intervals or a claim of statistical superiority.

Native ClearML screenshots have the UI axis label 'Iterations'; the pipeline reports epoch values. Panel loss scales differ. Curves support within-model training/validation analysis, not ranking models by raw loss.
