# ADR 0002: Two-Stage Figure Analysis with Sauvola Enhancement and Ground-Truth Callouts

## Status
Accepted

## Date
2026-10-20

## Context
Scanned engineering and scientific PDFs contain diverse visual figures ranging from continuous curves and mechanical cross-sections to block diagrams and schematics. The user requested merging `Fig2md` into `pdf2md` so that:
1. Convertible figures are transformed into native Mermaid diagrams and integrated into Obsidian notes.
2. Non-convertible figures have their image quality enhanced.
3. Base PDF-to-Markdown functionality remains intact.

We needed to decide how figures are routed, how diagrams and images are embedded in the generated Markdown note, and how image enhancement is stored.

## Decision
1. **Two-Stage Vision Routing**:
   - **Stage 1 (Classification)**: The cropped figure is analyzed by the vision model using a classification prompt to identify its archetype (`FLOWCHART_OR_SCHEMATIC`, `GRAPH_OR_PLOT`, `ENGINEERING_CROSS_SECTION`, etc.) and `is_mermaid_convertible` boolean.
   - **Stage 2 (Diagram Generation or Fallback)**:
     - If `is_mermaid_convertible` is `true`, a second vision prompt generates clean, syntactically valid Mermaid code.
     - The raster crop is also enhanced using fast $O(W \times H)$ integral-image Sauvola adaptive thresholding and morphological despeckle.
2. **Obsidian Markdown Output Layout**:
   - For **convertible figures**: Insert the native ```mermaid block, immediately followed by a collapsible Obsidian callout (`> [!figure]- Original Enhanced Image: <Label>`) containing the enhanced image link and caption. This provides immediate interactivity while retaining the ground-truth technical drawing for verification.
   - For **non-convertible figures**: Insert the enhanced image link and caption directly into the Markdown flow.
3. **Storage & Overwrite**:
   - The enhanced image is written directly to `images/<filename>` so that all Markdown references resolve to the cleaned, high-contrast monochrome version.
4. **Scope Control**:
   - Excalidraw generation from `Fig2md` is omitted to maintain a lean Obsidian-focused dependency footprint.
   - CLI flags `--no-mermaid` and `--no-figure-enhancement` allow disabling stages when desired.

## Consequences
### Positive
- Notes achieve modern interactive diagram rendering without sacrificing technical precision or loss of ground-truth data.
- Poorly scanned, noisy document figures are cleaned and readable.
- Fully compatible with standard Obsidian markdown viewers.

### Negative / Trade-offs
- Convertible figures require two vision model inference passes (classification + code synthesis), increasing token usage and API latency when processing diagram-heavy documents.

## Alternatives Considered
- **Replacing Image Entirely**: Omit raster image if Mermaid is generated. Rejected because technical diagrams may have nuanced labels or geometric details not captured by Mermaid.
- **Side-by-side or Non-collapsible Layout**: Always show both full-size. Rejected because it clutters the document flow when the Mermaid diagram is already primary.

