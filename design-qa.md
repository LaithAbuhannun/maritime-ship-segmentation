# Research portfolio visual review — 18 November 2026

final result: passed

## Current target

The user requested a cleaner, more beautiful site with better real dataset pictures and animations. The selected visual direction adapts Shakuro’s image-led Per Appelgren portfolio: restrained navigation, editorial typography, large photographs, whitespace and layered imagery. This is a research-specific adaptation, not an exact clone. Earlier dark dashboard styling is superseded.

## Evidence opened and compared

- Source: https://dribbble.com/shots/27824051-Website-Design-for-Fashion-and-Beauty-Photographer-Per-Appelgren
- Source capture: `/Users/laithhannun/Documents/Codex/dribbble-photography-reference.png` (1512 × 771 px).
- Revised desktop: `/Users/laithhannun/Documents/Codex/maritime-editorial-desktop.png` (1512 × 900 px; same CSS viewport width).
- Mobile: `/Users/laithhannun/Documents/Codex/maritime-editorial-mobile.png` (390 × 844 px).
- Combined comparison opened: `/Users/laithhannun/Documents/Codex/maritime-editorial-comparison.jpg`; proportional downsampling to adjacent 744-pixel columns. Different source framing, page content and heights are intentional; no exact geometric match claimed.

## Iteration findings

1. P2: overlapping floating panels obscured the hero caption. Moved source context to a separate footnote below the panels. Revised desktop capture shows the footnote clearly.
2. P2: the initial mobile title wrapped over three lines and pushed the imagery too far down. Reduced mobile title size and removed the redundant mobile byline; author information remains in the header, report and footer. Revised mobile capture shows the two-line title and main vessel photograph on the first screen.
3 Ambiguous initial HRSID illustration was replaced with the verified retained coastal image 1, a distinct filename to avoid stale browser cache, corrected alt text and provenance metadata. Browser confirms the harbour asset.

## Required visual surfaces

- Typography: system sans-serif for the research content; a restrained italic serif for the hero phrase; consistent readable body text, captions and technical labels. Clear heading hierarchy.
- Spacing/layout: image-led hero, generous whitespace, aligned three-column dataset gallery and two-column annotation comparison. Small screens stack sections. No document overflow at 1512 or 390 CSS pixels.
- Colours: warm off-white, forest green and sage; chart colours unchanged. Footer and source-code card hover colours adjusted for legibility. Dark report section deliberately retained as a distinct reading destination.
- Assets: original dataset photographs and retained SAR inputs/annotations. No AI-generated dataset pictures or reference artwork reused. New asset provenance in `docs/assets/dataset-image-sources.json`. Source valid-directory RGB examples labelled illustrative; ground-truth viewer explicitly not a prediction.
- Content: sourced maritime motivation, experiment results and report preserved. No new performance claims. Hero headline is presentation copy, not an operational deployment claim.

## Functional checks

- Native image dialog opens from a gallery card, shows its title/context, closes by button and Escape.
- Annotation range updates the clip and accessible percentage; ArrowRight produced 56%, End produced 100%.
- Dataset switch keyboard activation shows visible RF-DETR AP50:95 70.76 / AP50 90.95 / P 93.96 / R 86.15 / F1 89.88 / FPS 7.64.
- Prompted SAMSAR remains a separate optional row.
- Scroll reveals, card hover effects, lazy images and image transitions retained; reduced-motion CSS/JS guards present. OS preference switching was not manually exercised.
- No browser warning/error entries in local check.
- `python3 scripts/validate_portfolio.py`, `node --check docs/app.js`, and `git diff --check` passed.
- No GPU training or experimental metric regeneration performed.

## Follow-up polish

No outstanding P0/P1/P2 findings for this scope. Small dataset source images retain their native detail; they are not enhanced or synthesised.
