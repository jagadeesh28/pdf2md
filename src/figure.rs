use anyhow::{Context, Result, anyhow};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::GenericImageView;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigureDetection {
    #[serde(alias = "figure_id", alias = "label", alias = "name", alias = "title")]
    pub figure_number: String,
    #[serde(alias = "box", alias = "bbox", alias = "bounding_box")]
    pub box_2d: Vec<f32>, // [ymin, xmin, ymax, xmax] in 0..1000 or 0.0..1.0
}

#[derive(Debug, Clone)]
pub struct CroppedFigure {
    pub figure_label: String,
    pub filename: String,
    pub file_path: PathBuf,
}

#[derive(Debug, Serialize)]
struct VisionDetectionRequest {
    model: String,
    messages: Vec<VisionMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct VisionMessage {
    role: String,
    content: Vec<VisionContentPart>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum VisionContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlObject },
}

#[derive(Debug, Serialize)]
struct ImageUrlObject {
    url: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: Option<ChatMessage>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Option<String>,
}

const FIGURE_DETECTION_PROMPT: &str = r#"Detect all figures, charts, diagrams, drawings, and technical plots on this document page.
For each figure found, extract its exact figure number or label (for example: "Figure 1", "Figure 2.1", "Figure VII-3", "Fig. 4").
If a figure does not have an explicit number, use a concise identifier like "Figure (Untitled)".

Output ONLY a valid JSON array of objects in this exact schema:
[
  {
    "figure_number": "Figure VII-3",
    "box_2d": [ymin, xmin, ymax, xmax]
  }
]

Rules:
1. box_2d format: [ymin, xmin, ymax, xmax] normalized to integers from 0 to 1000 (0,0 is top-left, 1000,1000 is bottom-right).
2. The bounding box should cover the entire graphical illustration and diagram (and optionally its caption).
3. If no figures or plots are present on the page, output an empty JSON array: []
4. Return ONLY the raw JSON array. Never include markdown fences or explanatory text."#;

/// Detects figures on a document page image using Qwen Vision and crops each into output_dir.
pub fn detect_and_crop_figures(
    page_image_path: &Path,
    output_images_dir: &Path,
    page_num: usize,
) -> Result<Vec<CroppedFigure>> {
    fs::create_dir_all(output_images_dir).with_context(|| {
        format!(
            "failed to create figure output directory {}",
            output_images_dir.display()
        )
    })?;

    let img = image::open(page_image_path).with_context(|| {
        format!(
            "failed to open page image for cropping: {}",
            page_image_path.display()
        )
    })?;

    let detections = match detect_figures_with_qwen(page_image_path) {
        Ok(dets) => dets,
        Err(err) => {
            eprintln!(
                "Warning: Qwen figure detection on page {} encountered: {err:#}. Skipping figure cropping for this page.",
                page_num
            );
            return Ok(Vec::new());
        }
    };

    let mut cropped_figures = Vec::new();
    let (img_w, img_h) = img.dimensions();

    for (idx, det) in detections.into_iter().enumerate() {
        if det.box_2d.len() < 4 {
            continue;
        }

        let ymin = det.box_2d[0];
        let xmin = det.box_2d[1];
        let ymax = det.box_2d[2];
        let xmax = det.box_2d[3];

        // Normalize if in 0.0..1.0 range
        let scale = if ymax <= 1.0 && xmax <= 1.0 && (ymax > 0.0 || xmax > 0.0) {
            1000.0
        } else {
            1.0
        };

        let ymin_norm = (ymin * scale).clamp(0.0, 1000.0);
        let xmin_norm = (xmin * scale).clamp(0.0, 1000.0);
        let ymax_norm = (ymax * scale).clamp(0.0, 1000.0);
        let xmax_norm = (xmax * scale).clamp(0.0, 1000.0);

        if xmax_norm <= xmin_norm || ymax_norm <= ymin_norm {
            continue;
        }

        let px_x = ((xmin_norm / 1000.0) * img_w as f32).max(0.0) as u32;
        let px_y = ((ymin_norm / 1000.0) * img_h as f32).max(0.0) as u32;
        let px_w = (((xmax_norm - xmin_norm) / 1000.0) * img_w as f32)
            .min((img_w.saturating_sub(px_x)) as f32) as u32;
        let px_h = (((ymax_norm - ymin_norm) / 1000.0) * img_h as f32)
            .min((img_h.saturating_sub(px_y)) as f32) as u32;

        if px_w < 15 || px_h < 15 {
            continue;
        }

        let clean_label = sanitize_figure_label(&det.figure_number, page_num, idx + 1);
        let filename = format!("{}.png", sanitize_filename(&clean_label));
        let figure_path = output_images_dir.join(&filename);

        let cropped = img.crop_imm(px_x, px_y, px_w, px_h);
        cropped
            .save(&figure_path)
            .with_context(|| format!("failed to save cropped figure {}", figure_path.display()))?;

        println!(
            "Cropped figure '{}' -> {}",
            clean_label,
            figure_path.display()
        );

        cropped_figures.push(CroppedFigure {
            figure_label: clean_label,
            filename,
            file_path: figure_path,
        });
    }

    Ok(cropped_figures)
}

fn detect_figures_with_qwen(image_path: &Path) -> Result<Vec<FigureDetection>> {
    let bytes = fs::read(image_path).context("failed to read image file for detection")?;
    let image_base64 = STANDARD.encode(&bytes);

    let runtime = tokio::runtime::Runtime::new()
        .context("failed to create Tokio runtime for Qwen figure detection")?;

    runtime.block_on(async {
        let base_url = std::env::var("QWEN_URL")
            .or_else(|_| std::env::var("QWEN_API_BASE"))
            .or_else(|_| std::env::var("LOCAL_LLM_URL"))
            .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());

        let client = reqwest::Client::new();
        let model = crate::qwen::resolve_model_name(&client, &base_url).await;
        let api_key = std::env::var("QWEN_API_KEY").unwrap_or_default();

        let endpoint = if base_url.ends_with("/chat/completions") {
            base_url.clone()
        } else if base_url.ends_with("/v1") {
            format!("{base_url}/chat/completions")
        } else {
            format!("{}/v1/chat/completions", base_url.trim_end_matches('/'))
        };

        let payload = VisionDetectionRequest {
            model: model.clone(),
            messages: vec![VisionMessage {
                role: "user".to_string(),
                content: vec![
                    VisionContentPart::Text {
                        text: FIGURE_DETECTION_PROMPT.to_string(),
                    },
                    VisionContentPart::ImageUrl {
                        image_url: ImageUrlObject {
                            url: format!("data:image/png;base64,{image_base64}"),
                        },
                    },
                ],
            }],
            temperature: Some(0.1),
        };

        let mut request = client.post(&endpoint).json(&payload);
        if !api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request
            .send()
            .await
            .with_context(|| format!("failed to send detection request to {endpoint}"))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("failed to read detection response body")?;

        if !status.is_success() {
            return Err(anyhow!(
                "Qwen figure detection returned HTTP {status}: {body}"
            ));
        }

        let parsed: OpenAiChatResponse = serde_json::from_str(&body)
            .with_context(|| format!("failed to parse detection response: {body}"))?;

        let raw_content = parsed
            .choices
            .first()
            .and_then(|c| c.message.as_ref())
            .and_then(|m| m.content.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("[]");

        parse_detection_json(raw_content)
    })
}

pub fn parse_detection_json(raw_content: &str) -> Result<Vec<FigureDetection>> {
    let trimmed = raw_content.trim();

    // Remove markdown code fences if present
    let json_str = if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            &trimmed[start..=end]
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    let detections: Vec<FigureDetection> = serde_json::from_str(json_str)
        .with_context(|| format!("failed to parse figure detections JSON: {json_str}"))?;

    Ok(detections)
}

pub fn sanitize_figure_label(raw: &str, page_num: usize, index: usize) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("figure") {
        return format!("Figure P{page_num}-{index}");
    }

    // Capitalize standard prefix
    let lower = trimmed.to_lowercase();
    if lower.starts_with("fig.") || lower.starts_with("fig ") {
        format!("Figure {}", trimmed[4..].trim())
    } else if !lower.starts_with("figure") {
        format!("Figure {trimmed}")
    } else {
        trimmed.to_string()
    }
}

pub fn sanitize_filename(label: &str) -> String {
    let mut out = String::new();
    for c in label.chars() {
        if c.is_alphanumeric() || c == '-' || c == '_' {
            out.push(c);
        } else if (c.is_whitespace() || c == '.' || c == ':') && !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "Figure".to_string()
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_detection_json() {
        let json_data = r#"[
            {"figure_number": "Figure VII-3", "box_2d": [100, 150, 600, 850]},
            {"figure_number": "Fig. 4", "box_2d": [700, 200, 950, 800]}
        ]"#;

        let res = parse_detection_json(json_data).unwrap();
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].figure_number, "Figure VII-3");
        assert_eq!(res[0].box_2d, vec![100.0, 150.0, 600.0, 850.0]);
    }

    #[test]
    fn test_parse_detection_with_markdown_fences() {
        let fenced = "```json\n[\n  {\"figure_number\": \"Figure 1\", \"box_2d\": [50, 50, 400, 400]}\n]\n```";
        let res = parse_detection_json(fenced).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].figure_number, "Figure 1");
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("Figure VII-3"), "Figure-VII-3");
        assert_eq!(sanitize_filename("Figure 1.2: Orbit"), "Figure-1-2-Orbit");
        assert_eq!(sanitize_filename("Figure 4"), "Figure-4");
    }

    #[test]
    fn test_sanitize_figure_label() {
        assert_eq!(sanitize_figure_label("VII-3", 1, 1), "Figure VII-3");
        assert_eq!(sanitize_figure_label("Fig. 4", 1, 1), "Figure 4");
        assert_eq!(sanitize_figure_label("Figure 2.1", 1, 1), "Figure 2.1");
        assert_eq!(sanitize_figure_label("", 3, 1), "Figure P3-1");
    }
}
