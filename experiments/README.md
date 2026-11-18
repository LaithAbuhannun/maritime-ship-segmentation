# Code map

These are retained research implementations, not a newly unified package. Install one environment at a time: SAR and visible code both expose a package named `sarstudy`.

| Path | Purpose |
|---|---|
| `sar/source_v5_v6` | Source fingerprint matched to the selected SAR YOLO26, RF-DETR, SAMSAR and Ship Contour runs |
| `sar/source_v10_v11` | Source matched to the selected SwinInsSeg runs and the unsuccessful SAR SOLOv2 diagnostic runs |
| `sar/recorded_run_configs` | Saved per-run configurations; these take precedence over archived defaults |
| `visible` | Visible-light training, dataset preparation, evaluation and logging code retained for the final report |
| `../notebooks/sar` and `../notebooks/visible` | Colab workflow entrypoints; retained outputs are kept where available |

`results/source_verification.json` records the historical SAR fingerprint checks. Notebook outputs alone do not prove the provenance of every historical run. Do not mix an archived source tree with another run's configuration and claim an exact reproduction. Paths inside saved configurations refer to the original Colab/Drive layout and must be adapted for a new environment.

SAR SOLOv2 diagnostic runs produced zero AP; the later corrected rerun code is not used to overwrite those results. Paper-guided adapters have not been independently verified as exact paper reproductions. The external SAMSAR implementation and all weights are excluded.
