# Glossary: pdf2md Ubiquitous Language

This glossary establishes canonical terminology across `pdf2md` and its merged figure analysis pipeline (derived from `Fig2md`).

---

### Figure Archetype
The high-level structural classification of an extracted figure determined by a vision language model (Stage 1). Archetypes include:
- `FLOWCHART_OR_SCHEMATIC`: Block diagrams, logic flows, piping/fluid circuits, or electrical schematics.
- `GRAPH_OR_PLOT`: 2D or 3D coordinate charts, curves, nomographs, and data plots.
- `ENGINEERING_CROSS_SECTION`: 2D CAD sectional drawings with internal mechanical components, hatching, or numbered balloons.
- `ISOMETRIC_CUTAWAY`: 3D pictorial cutaways showing internal machinery.
- `TABLE_OR_TEXT`: Data tables or typographical snippets misclassified as figures.

### Mermaid Conversion
The process of transforming discrete structural diagrams (primarily `FLOWCHART_OR_SCHEMATIC`) into valid, executable Mermaid syntax code blocks (```mermaid ... ```) using vision model prompting.

### Adaptive Binarization (Sauvola)
A localized thresholding algorithm optimized using integral images (summed-area tables) running in $O(W \times H)$ time. It dynamically computes local mean and standard deviation to separate foreground lines and text from degraded, yellowed, or non-uniform background paper in scanned documents.

### Morphological Despeckle
A post-binarization filtering step that eliminates isolated 1-2 pixel noise artifacts ("salt and pepper" scanning dust) without eroding thin technical drawing lines or arrowheads.

### Enhanced Image
A cropped figure image that has passed through adaptive binarization and morphological despeckling, resulting in crisp monochrome line art with significantly reduced file size and improved visual clarity.

### Ground-Truth Fallback Callout
An Obsidian Markdown collapsible callout (`> [!figure]- Original Enhanced Image: Figure X`) placed directly below a generated Mermaid diagram containing the enhanced raster image and verbatim caption. It guarantees fidelity by allowing readers to cross-reference the original drawing against the AI-generated diagram.

### Two-Stage Vision Figure Router
A pipeline design separating figure understanding into two distinct steps:
1. **Stage 1 (Classification & Convertibility)**: Fast assessment of archetype and determination of whether the figure can be faithfully converted to a graph/Mermaid representation.
2. **Stage 2 (Diagram Synthesis or Enhancement Fallback)**: If convertible, generates Mermaid code; if not convertible, applies image enhancement to the raster crop.

