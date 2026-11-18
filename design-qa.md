# Portfolio design review — 18 November 2026

final result: passed

## Target and evidence

User request: choose suitable Dribbble inspiration and apply it to the existing research portfolio. This is an adaptation of the visual direction, not a pixel-for-pixel dashboard clone.

- Source: [George Railean, Geospatial Intelligence Dashboard & Mission Control UI](https://dribbble.com/shots/27894830-Geospatial-Intelligence-Dashboard-Mission-Control-UI).
- Source capture: `/Users/laithhannun/Documents/Codex/dribbble-geospatial-reference.png` (1512 × 771 pixels).
- Desktop implementation: `/Users/laithhannun/Documents/Codex/maritime-portfolio-desktop-v3.png` (1512 × 900 pixels; viewport 1512 × 900 CSS pixels).
- Mobile implementation: `/Users/laithhannun/Documents/Codex/maritime-portfolio-mobile-v3.png` (390 × 844 pixels; viewport 390 × 844 CSS pixels).
- Combined comparison opened and reviewed: `/Users/laithhannun/Documents/Codex/maritime-portfolio-design-comparison.jpg`. Captures downsampled proportionally to adjacent 744-pixel-wide columns. Different source and implementation content and heights are intentional; no exact geometry matching claimed.

## Findings and corrections

- P2 in first local capture: duplicate experiment label and nested hero wrapper. Removed duplicated wrapper/label and scroll cue. Latest desktop capture shows a single label, correct two-column structure and no clipping.
- No outstanding P0/P1/P2 findings in the revised desktop/mobile views.

## Required surfaces

- Typography: replaces editorial serif with a restrained system sans-serif, generous display heading and compact technical labels. Main title wraps intentionally over three desktop lines. Body remains legible and has a clear hierarchy.
- Spacing/layout: consistent section spacing, bounded cards, matched radii, substantial image area. Mobile stacks the hero and evidence grid; document width measured 390 at a 390-pixel viewport.
- Colours: charcoal base, blue-grey panels and pale cyan controls adapt the source direction. Charts retain their original evidence colours within light image panels; no data or curve colours are modified.
- Images: actual retained SAR/visible comparisons and native ClearML exports, preserving aspect ratios. No satellite illustration, fake charts or generated research images added.
- Content: existing methodological limitations and prompted SAMSAR distinction retained. Highlighted RF-DETR visible-light AP50:95 is 70.76%, matching the selected metrics. Report PDF unmodified.

## Browser checks

- Result dataset switch: visible-light table displays RF-DETR 70.76 AP50:95 / 90.95 AP50 / 93.96 precision / 86.15 recall / 89.88 F1 / 7.64 FPS.
- SAMSAR checkbox displays the prompted row separately.
- Qualitative dataset controls and visible-light training-history disclosure exercised.
- Navigation anchors and report links retained; PDF preview remains lazy-loaded.
- Desktop and mobile overflow checks passed.
- Browser error/warning log returned no entries for the local preview.
- Reduced-motion media rule suppresses animations and transforms; motion preference guard in JavaScript retained. OS preference switching not manually exercised.
- `node --check docs/app.js` and `python3 scripts/validate_portfolio.py` passed. No GPU training repeated.

## Follow-up polish

None required for this scoped redesign. Source footage is a dashboard; this site intentionally remains a portfolio with readable narrative and a report download.
