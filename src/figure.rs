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
    #[serde(default, alias = "figure_caption", alias = "desc", alias = "description")]
    pub caption: Option<String>,
    #[serde(alias = "box", alias = "bbox", alias = "bounding_box")]
    pub box_2d: BoundingBoxCoords,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BoundingBoxCoords {
    Array(Vec<f32>),
    Map {
        #[serde(alias = "top", alias = "y1")]
        ymin: f32,
        #[serde(alias = "left", alias = "x1")]
        xmin: f32,
        #[serde(alias = "bottom", alias = "y2")]
        ymax: f32,
        #[serde(alias = "right", alias = "x2")]
        xmax: f32,
    },
}

impl BoundingBoxCoords {
    /// Returns coordinates as `(xmin, ymin, xmax, ymax)`.
    pub fn to_coords(&self) -> Option<(f32, f32, f32, f32)> {
        match self {
            BoundingBoxCoords::Array(arr) => {
                if arr.len() < 4 {
                    None
                } else {
                    Some((arr[0], arr[1], arr[2], arr[3]))
                }
            }
            BoundingBoxCoords::Map { ymin, xmin, ymax, xmax } => {
                Some((*xmin, *ymin, *xmax, *ymax))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CroppedFigure {
    pub figure_label: String,
    pub caption: Option<String>,
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

const FIGURE_DETECTION_PROMPT: &str = r#"You are an expert document vision analyzer.
Your task is to detect ONLY actual, non-convertible visual figures and graphics on this page (such as plots, charts, graphs, flowcharts, schematics, engineering drawings, diagrams, and photos).

CRITICAL EXTRACTION RULES:
1. ONLY NON-CONVERTIBLE VISUAL GRAPHICS:
   - Crop ONLY actual visual drawings, charts, schematics, and plots that CANNOT be represented as Markdown text or Markdown tables.
   - STRICTLY FORBIDDEN: NEVER detect or crop text paragraphs, tables of contents, "List of Figures" index pages, bibliographies, or plain textual listings. If a page only contains text or a list of figure titles, return an empty array [].
2. INCLUDE BOTH THE GRAPHIC AND THE FULL FIGURE NUMBER AND CAPTION IN THE CROPPED BOX:
   - The bounding box [xmin, ymin, xmax, ymax] MUST enclose BOTH the visual graphic (plot/drawing/chart) AND its complete figure number and caption (e.g., "Figure 4.5.—Variation of species mole fractions...").
   - ymin MUST be the top of the graphic canvas, top border line, axis labels, or title (whichever is highest).
   - ymax MUST be the bottom of the figure caption text line(s).
   - The resulting cropped image MUST contain both the visual graphic and the readable figure number & caption text.
   - STOP IMMEDIATELY after the figure caption. Do NOT include any body paragraphs, explanatory text, or section headings below or above the caption!
3. EXACT BOUNDARIES:
   - xmin: Leftmost boundary of the graphic, axes, or title.
   - ymin: Topmost boundary of the graphic canvas, top border line, axis labels, or title (whichever is highest). Do NOT clip the top border line or uppermost curve.
   - xmax: Rightmost boundary of the graphic, legend, or caption.
   - ymax: Bottommost baseline of the figure caption. STRICTLY exclude body paragraphs beneath it.
   - Do NOT include surrounding page body paragraphs, page headers, or page footers.
4. Output format: Return ONLY a valid JSON array of objects in this exact schema:
[
  {
    "figure_number": "Figure 4.7",
    "caption": "Variation of species mole fractions with distance for shock tube problem.",
    "box_2d": [xmin, ymin, xmax, ymax]
  }
]
- Coordinates: [xmin, ymin, xmax, ymax] (0,0 is top-left).
- If no actual visual drawings, plots, or diagrams exist on this page, return []
- Return ONLY raw JSON without markdown fences or commentary."#;

/// Detects figures on a document page image using Qwen Vision, refines boundaries tightly,
/// and crops each into output_dir without overwriting existing files.
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
        let (mut xmin, mut ymin, mut xmax, mut ymax) = match det.box_2d.to_coords() {
            Some(coords) => coords,
            None => continue,
        };

        // Ensure proper ordering if inverted
        if xmin > xmax {
            std::mem::swap(&mut xmin, &mut xmax);
        }
        if ymin > ymax {
            std::mem::swap(&mut ymin, &mut ymax);
        }

        // Determine coordinate scale:
        // - 0.0..1.0 normalized
        // - 0..1000 standard Qwen-VL normalized
        // - Direct pixel coordinates if greater than 1000
        let (px_x, px_y, px_w, px_h) = if ymax <= 1.0 && xmax <= 1.0 && (ymax > 0.0 || xmax > 0.0) {
            let x = ((xmin * img_w as f32).clamp(0.0, img_w as f32)) as u32;
            let y = ((ymin * img_h as f32).clamp(0.0, img_h as f32)) as u32;
            let w = (((xmax - xmin) * img_w as f32).max(0.0) as u32).min(img_w.saturating_sub(x));
            let h = (((ymax - ymin) * img_h as f32).max(0.0) as u32).min(img_h.saturating_sub(y));
            (x, y, w, h)
        } else if ymax > 1000.0 || xmax > 1000.0 {
            // Direct pixel coordinates
            let x = (xmin.clamp(0.0, img_w as f32)) as u32;
            let y = (ymin.clamp(0.0, img_h as f32)) as u32;
            let w = ((xmax - xmin).max(0.0) as u32).min(img_w.saturating_sub(x));
            let h = ((ymax - ymin).max(0.0) as u32).min(img_h.saturating_sub(y));
            (x, y, w, h)
        } else {
            // 0..1000 range
            let ymin_norm = ymin.clamp(0.0, 1000.0);
            let xmin_norm = xmin.clamp(0.0, 1000.0);
            let ymax_norm = ymax.clamp(0.0, 1000.0);
            let xmax_norm = xmax.clamp(0.0, 1000.0);

            if xmax_norm <= xmin_norm || ymax_norm <= ymin_norm {
                continue;
            }

            let x = ((xmin_norm / 1000.0) * img_w as f32).max(0.0) as u32;
            let y = ((ymin_norm / 1000.0) * img_h as f32).max(0.0) as u32;
            let w = (((xmax_norm - xmin_norm) / 1000.0) * img_w as f32)
                .min((img_w.saturating_sub(x)) as f32) as u32;
            let h = (((ymax_norm - ymin_norm) / 1000.0) * img_h as f32)
                .min((img_h.saturating_sub(y)) as f32) as u32;
            (x, y, w, h)
        };

        // Filter out tiny slivers or text lines:
        // A visual figure must have substantial dimensions (at least 60x60 px or min area).
        // Slender horizontal text strips (e.g., width 150, height 30, or vice versa) from TOC/index lists must be rejected.
        if px_w < 60 || px_h < 60 || (px_w * px_h) < 5000 {
            println!(
                "Skipping candidate '{}' on page {page_num}: dimensions too small for a visual graphic ({}x{})",
                det.figure_number, px_w, px_h
            );
            continue;
        }

        // Tightly refine boundaries to content (trimming excess blank borders and ensuring clean edges)
        let (crop_x, crop_y, crop_w, crop_h) = refine_crop_boundaries(&img, px_x, px_y, px_w, px_h);

        // Verify refined crop meets minimum visual figure dimensions
        if crop_w < 50 || crop_h < 50 {
            continue;
        }

        let clean_label = sanitize_figure_label(&det.figure_number, page_num, idx + 1);
        let (filename, figure_path) = resolve_unique_figure_path(output_images_dir, &clean_label);

        let cropped = img.crop_imm(crop_x, crop_y, crop_w, crop_h);
        cropped
            .save(&figure_path)
            .with_context(|| format!("failed to save cropped figure {}", figure_path.display()))?;

        println!(
            "Cropped figure '{}' [box: x={}, y={}, w={}, h={}] -> {}",
            clean_label,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            figure_path.display()
        );

        cropped_figures.push(CroppedFigure {
            figure_label: clean_label,
            caption: det.caption.clone(),
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
    let lower = trimmed.to_lowercase();
    if trimmed.is_empty()
        || lower == "figure"
        || lower == "fig"
        || lower == "figure (untitled)"
        || lower == "untitled"
        || lower == "figure untitled"
        || lower == "none"
        || lower == "figure: untitled"
    {
        return format!("Figure P{page_num}-{index}");
    }

    // Capitalize standard prefix
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

/// Disambiguates figure filename so existing files in output_images_dir are never overwritten.
pub fn resolve_unique_figure_path(
    output_images_dir: &Path,
    clean_label: &str,
) -> (String, PathBuf) {
    let base_name = sanitize_filename(clean_label);
    let default_filename = format!("{base_name}.png");
    let default_path = output_images_dir.join(&default_filename);

    if !default_path.exists() {
        return (default_filename, default_path);
    }

    let mut counter = 1;
    loop {
        let candidate_filename = format!("{base_name}_{counter}.png");
        let candidate_path = output_images_dir.join(&candidate_filename);
        if !candidate_path.exists() {
            return (candidate_filename, candidate_path);
        }
        counter += 1;
    }
}

/// Snaps bounding box tightly around visual graphic content by trimming surrounding
/// white/blank space and adding a clean, uniform margin.
pub fn refine_crop_boundaries(
    img: &image::DynamicImage,
    px_x: u32,
    px_y: u32,
    px_w: u32,
    px_h: u32,
) -> (u32, u32, u32, u32) {
    let (img_w, img_h) = img.dimensions();
    if px_w == 0 || px_h == 0 {
        return (px_x, px_y, px_w, px_h);
    }

    // Expand search area slightly (3.5% of dimension, min 15px, max 45px) to ensure no top borders,
    // axis labels, superscript titles, or caption descenders are clipped even if the model's box was slightly tight.
    let margin_x = ((img_w as f32) * 0.035).clamp(15.0, 45.0) as u32;
    let margin_y = ((img_h as f32) * 0.035).clamp(15.0, 45.0) as u32;

    let search_x1 = px_x.saturating_sub(margin_x);
    let search_y1 = px_y.saturating_sub(margin_y);
    let search_x2 = (px_x + px_w + margin_x).min(img_w);
    let search_y2 = (px_y + px_h + margin_y).min(img_h);

    if search_x2 <= search_x1 || search_y2 <= search_y1 {
        return (px_x, px_y, px_w, px_h);
    }

    let rgba = img.to_rgba8();

    // Estimate background luminance from the boundary border pixels of the search region
    let mut bg_samples = Vec::new();
    for x in (search_x1..search_x2).step_by(5) {
        let p_top = rgba.get_pixel(x, search_y1);
        let p_bot = rgba.get_pixel(x, search_y2.saturating_sub(1));
        bg_samples.push(luminance(p_top));
        bg_samples.push(luminance(p_bot));
    }
    for y in (search_y1..search_y2).step_by(5) {
        let p_left = rgba.get_pixel(search_x1, y);
        let p_right = rgba.get_pixel(search_x2.saturating_sub(1), y);
        bg_samples.push(luminance(p_left));
        bg_samples.push(luminance(p_right));
    }

    let bg_luma = if !bg_samples.is_empty() {
        bg_samples.sort_unstable();
        bg_samples[bg_samples.len() * 3 / 4]
    } else {
        255
    };

    // A pixel is content if its luminance noticeably differs from background (threshold of 18)
    let is_content = |x: u32, y: u32| -> bool {
        let p = rgba.get_pixel(x, y);
        if p[3] < 30 {
            return false; // Transparent
        }
        let lum = luminance(p);
        if bg_luma >= 128 {
            lum < bg_luma.saturating_sub(18)
        } else {
            lum > bg_luma.saturating_add(18)
        }
    };

    let min_content_pixels = 2; // Avoid single-pixel noise/scanner artifacts

    // Identify continuous content bands and blank gaps in vertical search range
    #[derive(Debug, Clone, Copy)]
    struct ContentBand {
        start: u32,
        end: u32,
    }

    let mut bands: Vec<ContentBand> = Vec::new();
    let mut current_band_start = None;
    let mut consecutive_blanks = 0u32;
    // Minimum blank vertical space in rendered pixels separating a figure/caption from body text
    let blank_gap_threshold = ((search_y2.saturating_sub(search_y1)) as f32 * 0.02).clamp(10.0, 30.0) as u32;

    for y in search_y1..search_y2 {
        let count = (search_x1..search_x2)
            .filter(|&x| is_content(x, y))
            .take(min_content_pixels)
            .count();
        if count >= min_content_pixels {
            if current_band_start.is_none() {
                current_band_start = Some(y);
            }
            consecutive_blanks = 0;
        } else {
            consecutive_blanks += 1;
            if consecutive_blanks >= blank_gap_threshold {
                if let Some(start) = current_band_start {
                    let end = y.saturating_sub(consecutive_blanks);
                    if end >= start {
                        bands.push(ContentBand { start, end });
                    }
                    current_band_start = None;
                }
            }
        }
    }
    if let Some(start) = current_band_start {
        let end = search_y2.saturating_sub(consecutive_blanks);
        if end >= start {
            bands.push(ContentBand { start, end });
        }
    }

    if bands.is_empty() {
        return (px_x, px_y, px_w, px_h);
    }

    // Drop preceding bands caught solely by margin_y expansion:
    // If a band ends well above px_y (outside the model's detected box), it belongs to
    // preceding body text above the candidate box.
    while bands.len() > 1 && bands[0].end <= px_y.saturating_sub(6) {
        bands.remove(0);
    }

    // Drop trailing bands caught solely by margin_y expansion:
    // If a band starts well after px_y + px_h, it belongs to following body text below the candidate box.
    while bands.len() > 1 && bands.last().unwrap().start >= (px_y + px_h).saturating_add(6) {
        bands.pop();
    }

    // Drop trailing body text paragraph:
    // If the candidate box extended into a multi-line body paragraph beneath the figure/caption
    // (height >= 100px after substantial content >= 60px), truncate before that trailing paragraph.
    while bands.len() > 1 {
        let last_idx = bands.len() - 1;
        let last_h = bands[last_idx].end.saturating_sub(bands[last_idx].start);
        let prev_h = bands[last_idx - 1].end.saturating_sub(bands[0].start);
        if last_h >= 100 && prev_h >= 60 {
            bands.pop();
        } else {
            break;
        }
    }

    let content_y1 = bands[0].start;
    let content_y2 = bands.last().unwrap().end;

    // Scan vertical columns from left to find start of content
    let mut content_x1 = None;
    for x in search_x1..search_x2 {
        let count = (content_y1..=content_y2)
            .filter(|&y| is_content(x, y))
            .take(min_content_pixels)
            .count();
        if count >= min_content_pixels {
            content_x1 = Some(x);
            break;
        }
    }

    // Scan vertical columns from right to find end of content
    let mut content_x2 = None;
    for x in (search_x1..search_x2).rev() {
        let count = (content_y1..=content_y2)
            .filter(|&y| is_content(x, y))
            .take(min_content_pixels)
            .count();
        if count >= min_content_pixels {
            content_x2 = Some(x);
            break;
        }
    }

    match (content_x1, content_x2) {
        (Some(cx1), Some(cx2)) if cx2 > cx1 && content_y2 > content_y1 => {
            // Add a clean, consistent 14-pixel margin around graphic so top/bottom/side edges are never clipped
            let pad = 14u32;
            let final_x = cx1.saturating_sub(pad);
            let final_y = content_y1.saturating_sub(pad);
            let final_x2 = (cx2 + pad + 1).min(img_w);
            let final_y2 = (content_y2 + pad + 1).min(img_h);
            let final_w = final_x2.saturating_sub(final_x);
            let final_h = final_y2.saturating_sub(final_y);

            if final_w >= 15 && final_h >= 15 {
                (final_x, final_y, final_w, final_h)
            } else {
                (px_x, px_y, px_w, px_h)
            }
        }
        _ => (px_x, px_y, px_w, px_h),
    }
}

#[inline]
fn luminance(p: &image::Rgba<u8>) -> u8 {
    ((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114) / 1000) as u8
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
        assert_eq!(res[0].box_2d.to_coords(), Some((100.0, 150.0, 600.0, 850.0)));
    }

    #[test]
    fn test_parse_detection_with_map_coords() {
        let json_data = r#"[
            {"figure_number": "Figure 1", "box_2d": {"ymin": 50, "xmin": 60, "ymax": 400, "xmax": 500}}
        ]"#;
        let res = parse_detection_json(json_data).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].figure_number, "Figure 1");
        assert_eq!(res[0].box_2d.to_coords(), Some((60.0, 50.0, 500.0, 400.0)));
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
        assert_eq!(sanitize_figure_label("Figure (Untitled)", 3, 2), "Figure P3-2");
        assert_eq!(sanitize_figure_label("untitled", 4, 1), "Figure P4-1");
    }

    #[test]
    fn test_resolve_unique_figure_path() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path();

        let (name1, p1) = resolve_unique_figure_path(path, "Figure 1");
        assert_eq!(name1, "Figure-1.png");
        fs::write(&p1, b"dummy").unwrap();

        let (name2, p2) = resolve_unique_figure_path(path, "Figure 1");
        assert_eq!(name2, "Figure-1_1.png");
        fs::write(&p2, b"dummy2").unwrap();

        let (name3, _) = resolve_unique_figure_path(path, "Figure 1");
        assert_eq!(name3, "Figure-1_2.png");
    }

    #[test]
    fn test_refine_crop_boundaries_trims_whitespace() {
        use image::{Rgba, RgbaImage};

        // Create a 200x200 white image
        let mut img_buf = RgbaImage::from_pixel(200, 200, Rgba([255, 255, 255, 255]));

        // Draw a black content box at (50, 60) of size (80, 70) -> bounds: x=50..130, y=60..130
        for y in 60..130 {
            for x in 50..130 {
                img_buf.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
        let dyn_img = image::DynamicImage::ImageRgba8(img_buf);

        // Suppose Qwen returned a loose box: x=30, y=40, w=120, h=110
        let (rx, ry, rw, rh) = refine_crop_boundaries(&dyn_img, 30, 40, 120, 110);

        // With 14px padding:
        // x should be 50 - 14 = 36
        // y should be 60 - 14 = 46
        // x2 should be 129 + 14 + 1 = 144 -> w = 144 - 36 = 108
        // y2 should be 129 + 14 + 1 = 144 -> h = 144 - 46 = 98
        assert_eq!(rx, 36);
        assert_eq!(ry, 46);
        assert_eq!(rw, 108);
        assert_eq!(rh, 98);
    }

    #[test]
    fn test_parse_detection_with_caption() {
        let json_data = r#"[
            {
                "figure_number": "Figure 4.7",
                "caption": "Variation of species mole fractions with distance.",
                "box_2d": [10, 20, 100, 200]
            }
        ]"#;
        let res = parse_detection_json(json_data).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].figure_number, "Figure 4.7");
        assert_eq!(
            res[0].caption.as_deref(),
            Some("Variation of species mole fractions with distance.")
        );
    }

    #[test]
    fn test_refine_crop_preserves_separated_caption() {
        use image::{Rgba, RgbaImage};

        // Create a 300x300 white canvas
        let mut img_buf = RgbaImage::from_pixel(300, 300, Rgba([255, 255, 255, 255]));

        // Block 1: Graphic from y=20..150
        for y in 20..150 {
            for x in 30..250 {
                img_buf.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }

        // Blank vertical gap between graphic and caption: y=150..170 (20px gap)

        // Block 2: Figure caption from y=170..195 (25px height <= 95px)
        for y in 170..195 {
            for x in 30..250 {
                img_buf.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }

        let dyn_img = image::DynamicImage::ImageRgba8(img_buf);
        let (_rx, ry, _rw, rh) = refine_crop_boundaries(&dyn_img, 10, 10, 260, 220);

        // Refined crop MUST include the caption (bottom should be past 195 + pad)
        assert_eq!(ry, 6); // 20 - 14pad
        assert!(ry + rh >= 200, "Refined bottom {} should enclose the caption (>= 200)", ry + rh);
    }

    #[test]
    fn test_refine_crop_separates_trailing_paragraph_text() {
        use image::{Rgba, RgbaImage};

        // Create a 300x400 white canvas
        let mut img_buf = RgbaImage::from_pixel(300, 400, Rgba([255, 255, 255, 255]));

        // Block 1: Graphic + caption from y=20..150
        for y in 20..150 {
            for x in 30..250 {
                img_buf.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }

        // Blank vertical gap from y=150..180 (30px gap > blank_gap_threshold)

        // Block 2: Trailing body paragraph from y=180..320
        for y in 180..320 {
            for x in 30..250 {
                img_buf.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }

        let dyn_img = image::DynamicImage::ImageRgba8(img_buf);
        let (_rx, ry, rw, rh) = refine_crop_boundaries(&dyn_img, 10, 10, 260, 350);

        // The refinement must detect the blank gap after y=150 and truncate before the trailing paragraph
        assert_eq!(ry, 6); // 20 - 14pad
        assert!(ry + rh <= 170, "Refined bottom {} should be at or before gap (<= 170)", ry + rh);
        assert!(rw >= 200, "Refined width {} should cover the content", rw);
    }

    #[test]
    fn test_refine_crop_preserves_multitier_flowchart_and_caption() {
        use image::{Rgba, RgbaImage};

        // Create a 350x500 white canvas
        let mut img_buf = RgbaImage::from_pixel(350, 500, Rgba([255, 255, 255, 255]));

        // Preceding paragraph text line: y=15..25 (above candidate box)
        for y in 15..25 {
            for x in 30..250 {
                img_buf.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
        // Blank gap from y=25..50

        // Flowchart Tier 1: y=50..110
        for y in 50..110 {
            for x in 60..240 {
                img_buf.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
        // Gap y=110..125

        // Flowchart Tier 2: y=125..180
        for y in 125..180 {
            for x in 60..240 {
                img_buf.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
        // Gap y=180..195

        // Flowchart Tier 3 (wider): y=195..260, x=30..280
        for y in 195..260 {
            for x in 30..280 {
                img_buf.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
        // Gap y=260..275

        // Figure Caption: y=275..295 (height 20px <= 95px)
        for y in 275..295 {
            for x in 40..270 {
                img_buf.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
        // Gap y=295..325

        // Trailing body paragraph: y=325..460 (height 135px >= 100px)
        for y in 325..460 {
            for x in 30..280 {
                img_buf.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }

        let dyn_img = image::DynamicImage::ImageRgba8(img_buf);
        // Candidate box detects the flowchart and caption: px_x=20, px_y=45, px_w=270, px_h=260 (y covers 45..305)
        let (rx, ry, rw, rh) = refine_crop_boundaries(&dyn_img, 20, 45, 270, 260);

        // 1. Preceding text at y=15..25 must be excluded: ry must start around 50 - 14pad = 36
        assert!(ry >= 35 && ry <= 40, "ry was {} (expected ~36, excluding preceding text at y=15..25)", ry);

        // 2. Trailing text at y=325..460 must be excluded: bottom must be before y=320
        assert!(ry + rh <= 320, "Bottom ry+rh was {} (expected <= 320, excluding trailing paragraph at y=325)", ry + rh);

        // 3. Caption at y=275..295 must be INCLUDED: bottom must be after 295 + 14pad = 309
        assert!(ry + rh >= 305, "Bottom ry+rh was {} (expected >= 305, enclosing caption up to y=295)", ry + rh);

        // 4. Width must cover the widest tier (x=30..280): rx <= 16, rx+rw >= 294
        assert!(rx <= 20, "rx was {} (expected <= 20 to cover x=30)", rx);
        assert!(rx + rw >= 290, "Right edge rx+rw was {} (expected >= 290 to cover x=280)", rx + rw);
    }
}
