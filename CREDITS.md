# Credits and sources

## Research and supervision
Research project by Laith Abuhannun, Bachelor of Computing Science (Honours), University of Technology Sydney, 2026. Supervisor: Dr Nabin Sharma. Thanks to Mahdi Rahimi for supplying the SAR images associated with his research and to comvis for the visible-light dataset.

## Datasets
- [SSDD official release](https://github.com/TianwenZhang0825/Official-SSDD) — Zhang et al., 2021, doi:10.3390/rs13183690.
- [HRSID](https://github.com/chaozhong2010/HRSID) — Wei et al., 2020, doi:10.1109/ACCESS.2020.3005861.
- [LS-SSDD-v1.0](https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN) — Zhang et al., 2020, doi:10.3390/rs12182997.
- [visible-ship-dataset, version 1](https://universe.roboflow.com/comvis-cekot/visible-ship-dataset) — comvis / Roboflow Universe, May 2024, CC BY 4.0 according to the retained export README. The original export has 3,124 images; subtypes were collapsed into one ship class.

Selected visual excerpts are included for research comparison and attributed here. Full datasets, private email correspondence, pretrained weights and trained checkpoints are not redistributed. Dataset copies and annotation versions supplied during the project may differ from current public downloads.

## Model families and software
This work implements, adapts and evaluates existing models. It does not introduce these architectures or claim exact independent reproduction of every published component.

- YOLO26-Seg: [Ultralytics](https://github.com/ultralytics/ultralytics).
- RF-DETR-Seg: [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).
- SAMSAR: Rahimi and Sharifian, 2025, [doi:10.1016/j.eswa.2025.127852](https://doi.org/10.1016/j.eswa.2025.127852); external reference source is required and is not redistributed. Uses [Segment Anything](https://github.com/facebookresearch/segment-anything).
- Ship Contour: Chen et al., 2025, [doi:10.3390/jmse13030519](https://doi.org/10.3390/jmse13030519).
- SwinInsSeg: Rabi Sharma et al., 2025, [doi:10.3390/math13010165](https://doi.org/10.3390/math13010165).
- SOLOv2-GALA: Sun et al., 2023, [doi:10.1371/journal.pone.0279248](https://doi.org/10.1371/journal.pone.0279248).
- Training and tracking use PyTorch, MMDetection/MMCV, pycocotools and ClearML. Third-party packages and implementations retain their original terms; this repository grants no new rights over them.

## Tool assistance
ChatGPT/Codex assisted with code development and debugging, drafting/reviewing text, figure and chart layout, LaTeX formatting, and preparation of this portfolio. Scientific charts are produced from retained numeric histories or native ClearML exports; qualitative images are retained model outputs, not generative illustrations. The portfolio chart is generated directly from the selected metric JSON. The author is responsible for reviewing the work and its conclusions.

## Portfolio design reference

The portfolio's dark surfaces, restrained cyan palette and layered evidence panels take visual inspiration from [Geospatial Intelligence Dashboard & Mission Control UI by George Railean](https://dribbble.com/shots/27894830-Geospatial-Intelligence-Dashboard-Mission-Control-UI). The implementation is an original adaptation for this research portfolio. Its satellite artwork and interface assets are not reused; the research images, curves and measurements are this project's retained experimental evidence.

The later image-led refresh also draws on Shakuro’s [Per Appelgren photography portfolio](https://dribbble.com/shots/27824051-Website-Design-for-Fashion-and-Beauty-Photographer-Per-Appelgren): restrained navigation, generous whitespace and image-first composition. No reference artwork is reused. Dataset image context is recorded in `docs/assets/dataset-image-sources.json`; source dataset illustrations are not presented as final-test predictions.
