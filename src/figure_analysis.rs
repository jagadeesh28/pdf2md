use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::str::FromStr;

use crate::backends::OcrBackend;
use crate::figure::CroppedFigure;
use crate::preprocess::ImagePreprocessor;

pub const STAGE1_CLASSIFICATION_PROMPT: &str = r#"You are an expert engineering archivist analyzing a scanned technical document figure.
Analyze the provided image and return ONLY a valid JSON object matching this schema:
{
  "figure_id": "string (e.g. 5.2.3.1a or 3.12 or null if not detected)",
  "archetype": "FLOWCHART_OR_SCHEMATIC | GRAPH_OR_PLOT | ENGINEERING_CROSS_SECTION | ISOMETRIC_CUTAWAY | TABLE_OR_TEXT",
  "is_composite": false,
  "secondary_archetypes": [],
  "composite_details": null,
  "is_mermaid_convertible": false,
  "mermaid_diagram_type": null,
  "caption": "string (verbatim caption text if visible, or a concise descriptive title)",
  "source_citation": "string (source citation or publication reference if printed on page)",
  "confidence": 0.95,
  "summary": "string (1-2 sentence description of what the figure depicts and its technical significance)"
}

Archetype definitions:
- FLOWCHART_OR_SCHEMATIC: Block diagrams, hydraulic/electrical schematics, process flowcharts with nodes, directed arrows, valves, switches, or logic.
- GRAPH_OR_PLOT: 2D or 3D coordinate plots, curves, nomographs, performance charts with X/Y axes, tick marks, legends.
- ENGINEERING_CROSS_SECTION: 2D mechanical cutaway or sectional CAD drawing with numbered callout balloons/pointers, hatching, internal mechanical components.
- ISOMETRIC_CUTAWAY: 3D perspective or isometric pictorial cutaway showing interior mechanical parts.
- TABLE_OR_TEXT: Data tables, equations, or pure text snippets.

MERMAID DIAGRAM CONVERSION DECISION:
- "is_mermaid_convertible": Set to true ONLY if the figure contains a discrete graph structure (flowchart, functional block diagram, electrical/fluid circuit schematic, state machine, or sequence flow) that can be accurately represented in Mermaid syntax.
- For all continuous curves, coordinate plots, mechanical cutaways, CAD drawings, photos, micrographs, or dense tables, set "is_mermaid_convertible": false.
- "mermaid_diagram_type": If is_mermaid_convertible is true, specify "flowchart", "stateDiagram", "sequenceDiagram", or "classDiagram". Otherwise set to null.

CRITICAL - COMPOSITE / COMBINATION FIGURES:
Technical documents frequently feature figures that combine multiple representations:
- Drawings inside plots: A coordinate plot/graph with an inset mechanical cross-section, physical drawing, or nozzle/valve geometry.
- Multi-panel hybrid figures: e.g. panel (a) is a coordinate plot, panel (b) is a physical cutaway or hydraulic schematic.
- Schematics with embedded curves: e.g. a circuit or fluid circuit schematic with an adjacent or inset characteristic response curve.
When ANY combination of items is detected:
1. Set "archetype" to the PRIMARY / DOMINANT archetype.
2. Set "is_composite": true.
3. Add all additional present archetypes to "secondary_archetypes" (e.g. ["ENGINEERING_CROSS_SECTION"] or ["GRAPH_OR_PLOT"]).
4. Set "composite_details" to a concise description of the composition (e.g. "Main panel is a 2D coordinate plot of pressure gradient; inset lower section depicts physical cross-sectional drawing of fluid orifice with cavitation").
If the figure is NOT a combination, set "is_composite": false, "secondary_archetypes": [], and "composite_details": null.

Return ONLY the JSON object. Do not wrap in markdown or prose."#;

pub const STAGE2_MERMAID_PROMPT: &str = r#"You are an expert technical diagram analyst.
Analyze the provided schematic, flowchart, block diagram, or process flow diagram.
Generate a clean, syntactically valid Mermaid diagram representing this system.

Guidelines:
1. Choose the most appropriate Mermaid diagram type:
   - `graph LR` or `graph TD` for flowcharts, piping/fluid circuits, block diagrams, or logic circuits.
   - `sequenceDiagram` for sequence or message timing diagrams.
   - `stateDiagram-v2` for finite state machines or transition diagrams.
2. Quote all node labels containing special characters, parentheses, numbers, or symbols:
   e.g. `id["Valve (V-101)"]` or `step1["Check Pressure (P >= 50 psi)"]`.
3. Do not use HTML tags inside node labels or edge labels.
4. Preserve all directed connections, branchings, decision paths, and edge labels:
   e.g. `A -->|"Fuel Flow"| B` or `Check -->|"Yes"| Action`.
5. Return ONLY the valid Mermaid code block enclosed in ```mermaid and ```. Do not include introductory or concluding prose."#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FigureArchetype {
    FlowchartOrSchematic,
    GraphOrPlot,
    EngineeringCrossSection,
    IsometricCutaway,
    TableOrText,
}

impl std::fmt::Display for FigureArchetype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FlowchartOrSchematic => write!(f, "FLOWCHART_OR_SCHEMATIC"),
            Self::GraphOrPlot => write!(f, "GRAPH_OR_PLOT"),
            Self::EngineeringCrossSection => write!(f, "ENGINEERING_CROSS_SECTION"),
            Self::IsometricCutaway => write!(f, "ISOMETRIC_CUTAWAY"),
            Self::TableOrText => write!(f, "TABLE_OR_TEXT"),
        }
    }
}

impl FromStr for FigureArchetype {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().trim() {
            "FLOWCHART_OR_SCHEMATIC" | "FLOWCHART" | "SCHEMATIC" => {
                Ok(Self::FlowchartOrSchematic)
            }
            "GRAPH_OR_PLOT" | "GRAPH" | "PLOT" => Ok(Self::GraphOrPlot),
            "ENGINEERING_CROSS_SECTION" | "CROSS_SECTION" | "CAD" => {
                Ok(Self::EngineeringCrossSection)
            }
            "ISOMETRIC_CUTAWAY" | "ISOMETRIC" | "CUTAWAY" => Ok(Self::IsometricCutaway),
            "TABLE_OR_TEXT" | "TABLE" | "TEXT" => Ok(Self::TableOrText),
            other => Err(anyhow!("Unknown figure archetype: '{other}'")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigureMetadata {
    #[serde(default)]
    pub figure_id: Option<String>,
    pub archetype: FigureArchetype,
    #[serde(default)]
    pub is_composite: bool,
    #[serde(default)]
    pub secondary_archetypes: Vec<FigureArchetype>,
    #[serde(default)]
    pub composite_details: Option<String>,
    #[serde(default)]
    pub is_mermaid_convertible: bool,
    #[serde(default)]
    pub mermaid_diagram_type: Option<String>,
    #[serde(default)]
    pub caption: Option<String>,
    #[serde(default)]
    pub source_citation: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub summary: Option<String>,
}

impl FigureMetadata {
    pub fn parse_from_json(json_str: &str) -> Result<Self> {
        let trimmed = json_str.trim();
        let cleaned = if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                &trimmed[start..=end]
            } else {
                trimmed
            }
        } else {
            trimmed
        };

        serde_json::from_str(cleaned)
            .with_context(|| format!("Failed to parse FigureMetadata JSON: {cleaned}"))
    }

    pub fn can_convert_to_mermaid(&self) -> bool {
        self.is_mermaid_convertible || self.archetype == FigureArchetype::FlowchartOrSchematic
    }
}

/// Extract clean Mermaid markdown code fence from an LLM response string.
pub fn extract_mermaid_code(response: &str) -> String {
    let trimmed = response.trim();
    if let Some(start) = trimmed.find("```mermaid") {
        if let Some(end) = trimmed[start + 10..].find("```") {
            let inner = &trimmed[start..start + 10 + end + 3];
            return format!("{}\n", inner.trim());
        }
    } else if let Some(start) = trimmed.find("```") {
        if let Some(end) = trimmed[start + 3..].find("```") {
            let inner = &trimmed[start + 3..start + 3 + end];
            return format!("```mermaid\n{}\n```\n", inner.trim());
        }
    }

    if trimmed.starts_with("graph")
        || trimmed.starts_with("flowchart")
        || trimmed.starts_with("sequenceDiagram")
        || trimmed.starts_with("stateDiagram")
        || trimmed.starts_with("classDiagram")
    {
        return format!("```mermaid\n{}\n```\n", trimmed);
    }

    format!("```mermaid\n{}\n```\n", trimmed)
}

/// Query the active backend vision model for figure analysis or diagram generation.
pub fn query_backend_vision(
    backend: OcrBackend,
    image_path: &Path,
    prompt: &str,
    model_override: Option<&str>,
) -> Result<String> {
    match backend {
        OcrBackend::Qwen => {
            crate::backends::qwen::query_vision(image_path, prompt, model_override)
        }
        OcrBackend::Gemini => {
            let api_key = std::env::var("GEMINI_API_KEY").map_err(|_| {
                anyhow!("Gemini vision query requires GEMINI_API_KEY environment variable")
            })?;
            crate::backends::gemini::query_vision(&api_key, image_path, prompt, model_override)
        }
        OcrBackend::Gemma => {
            let api_key = std::env::var("GEMMA_API_KEY")
                .or_else(|_| std::env::var("GEMINI_API_KEY"))
                .map_err(|_| {
                    anyhow!(
                        "Gemma vision query requires GEMMA_API_KEY or GEMINI_API_KEY environment variable"
                    )
                })?;
            crate::backends::gemma::query_vision(&api_key, image_path, prompt, model_override)
        }
    }
}

/// Classify an extracted figure image into an archetype and check Mermaid convertibility.
pub fn classify_figure(
    backend: OcrBackend,
    image_path: &Path,
    model_override: Option<&str>,
) -> Result<FigureMetadata> {
    let raw_json =
        query_backend_vision(backend, image_path, STAGE1_CLASSIFICATION_PROMPT, model_override)?;
    FigureMetadata::parse_from_json(&raw_json)
}

/// Synthesize a clean Mermaid diagram code block for a convertible schematic or flowchart.
pub fn generate_mermaid(
    backend: OcrBackend,
    image_path: &Path,
    model_override: Option<&str>,
) -> Result<String> {
    let raw = query_backend_vision(backend, image_path, STAGE2_MERMAID_PROMPT, model_override)?;
    Ok(extract_mermaid_code(&raw))
}

/// Analyze and enhance a cropped figure:
/// 1. If enhancement is enabled (default), run Sauvola adaptive binarization + despeckle in place.
/// 2. If Mermaid is enabled (default), classify the figure; if convertible, generate Mermaid code.
pub fn analyze_and_enhance_figure(
    backend: OcrBackend,
    fig: &mut CroppedFigure,
) -> Result<()> {
    let enable_enhancement = std::env::var("PDF2MD_NO_ENHANCE")
        .map(|v| v != "1" && v.to_lowercase() != "true")
        .unwrap_or(true);

    let enable_mermaid = std::env::var("PDF2MD_NO_MERMAID")
        .map(|v| v != "1" && v.to_lowercase() != "true")
        .unwrap_or(true);

    let model_override = std::env::var("PDF2MD_FIGURE_MODEL")
        .ok()
        .filter(|s| !s.trim().is_empty());

    // Step 1: Image Enhancement (Sauvola adaptive binarization + despeckle)
    if enable_enhancement {
        if let Err(e) = ImagePreprocessor::enhance_image_file(&fig.file_path) {
            eprintln!(
                "Image enhancement warning for '{}' ({}): {e:#}",
                fig.figure_label,
                fig.file_path.display()
            );
        } else {
            println!(
                "Enhanced figure image '{}' -> {}",
                fig.figure_label,
                fig.file_path.display()
            );
        }
    }

    // Step 2: Mermaid Decision & Synthesis
    if enable_mermaid {
        match classify_figure(backend, &fig.file_path, model_override.as_deref()) {
            Ok(meta) => {
                if meta.can_convert_to_mermaid() {
                    println!(
                        "Figure '{}' classified as {} (convertible to Mermaid). Synthesizing diagram...",
                        fig.figure_label, meta.archetype
                    );
                    match generate_mermaid(backend, &fig.file_path, model_override.as_deref()) {
                        Ok(code) => {
                            println!(
                                "Generated Mermaid diagram for figure '{}'",
                                fig.figure_label
                            );
                            fig.mermaid_code = Some(code);
                        }
                        Err(e) => {
                            eprintln!(
                                "Mermaid generation failed for '{}': {e:#}. Preserving enhanced image only.",
                                fig.figure_label
                            );
                        }
                    }
                } else {
                    println!(
                        "Figure '{}' classified as {} (not convertible to Mermaid). Preserving enhanced image.",
                        fig.figure_label, meta.archetype
                    );
                }
            }
            Err(e) => {
                eprintln!(
                    "Figure classification skipped for '{}': {e:#}",
                    fig.figure_label
                );
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_archetype_parsing() {
        assert_eq!(
            "FLOWCHART_OR_SCHEMATIC".parse::<FigureArchetype>().unwrap(),
            FigureArchetype::FlowchartOrSchematic
        );
        assert_eq!(
            "flowchart".parse::<FigureArchetype>().unwrap(),
            FigureArchetype::FlowchartOrSchematic
        );
        assert_eq!(
            "GRAPH_OR_PLOT".parse::<FigureArchetype>().unwrap(),
            FigureArchetype::GraphOrPlot
        );
        assert_eq!(
            "plot".parse::<FigureArchetype>().unwrap(),
            FigureArchetype::GraphOrPlot
        );
        assert_eq!(
            "ENGINEERING_CROSS_SECTION"
                .parse::<FigureArchetype>()
                .unwrap(),
            FigureArchetype::EngineeringCrossSection
        );
    }

    #[test]
    fn test_parse_stage1_json_clean() {
        let json_str = r#"{
            "figure_id": "3.1",
            "archetype": "FLOWCHART_OR_SCHEMATIC",
            "is_composite": false,
            "secondary_archetypes": [],
            "composite_details": null,
            "is_mermaid_convertible": true,
            "mermaid_diagram_type": "flowchart",
            "caption": "Functional Block Diagram",
            "confidence": 0.95,
            "summary": "Block diagram of propulsion fuel feed system."
        }"#;

        let meta = FigureMetadata::parse_from_json(json_str).unwrap();
        assert_eq!(meta.archetype, FigureArchetype::FlowchartOrSchematic);
        assert!(meta.is_mermaid_convertible);
        assert!(meta.can_convert_to_mermaid());
        assert_eq!(meta.figure_id.as_deref(), Some("3.1"));
    }

    #[test]
    fn test_parse_stage1_json_wrapped_markdown() {
        let json_str = "```json\n{\n  \"archetype\": \"GRAPH_OR_PLOT\",\n  \"is_mermaid_convertible\": false\n}\n```";
        let meta = FigureMetadata::parse_from_json(json_str).unwrap();
        assert_eq!(meta.archetype, FigureArchetype::GraphOrPlot);
        assert!(!meta.is_mermaid_convertible);
        assert!(!meta.can_convert_to_mermaid());
    }

    #[test]
    fn test_extract_mermaid_code() {
        let raw = "Here is the diagram:\n```mermaid\ngraph LR\n    A[\"Start\"] --> B[\"Stop\"]\n```\nDone.";
        let extracted = extract_mermaid_code(raw);
        assert_eq!(
            extracted.trim(),
            "```mermaid\ngraph LR\n    A[\"Start\"] --> B[\"Stop\"]\n```"
        );

        let bare = "graph TD\n    A --> B";
        let extracted_bare = extract_mermaid_code(bare);
        assert_eq!(
            extracted_bare.trim(),
            "```mermaid\ngraph TD\n    A --> B\n```"
        );
    }
}

