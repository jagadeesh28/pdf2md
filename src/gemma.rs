use anyhow::{Context, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<Content>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Debug, Serialize)]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
struct Part {
    #[serde(rename = "text")]
    text: Option<String>,
    #[serde(rename = "inlineData")]
    inline_data: Option<InlineData>,
}

#[derive(Debug, Serialize)]
struct InlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: Option<ContentResponse>,
}

#[derive(Debug, Deserialize)]
struct ContentResponse {
    #[serde(default)]
    parts: Vec<PartResponse>,
}

#[derive(Debug, Deserialize)]
struct PartResponse {
    text: Option<String>,
    #[serde(default)]
    thought: Option<bool>,
}

pub fn ocr_image_file(api_key: &str, image_path: &Path) -> Result<String> {
    let bytes = fs::read(image_path).context("failed to read OCR image file")?;
    let image_base64 = STANDARD.encode(&bytes);

    let runtime =
        tokio::runtime::Runtime::new().context("failed to create Tokio runtime for Gemini OCR")?;

    runtime.block_on(async {
        let payload = GeminiRequest {
            contents: vec![Content {
                parts: vec![
                    Part {
                        text: Some(r#"You are an expert OCR and transcription engine specializing in technical engineering reports and aerospace publications. Your task is to transcribe the provided document page image into verbatim, publication-quality Obsidian Markdown.

### 1. Document Structure & Preamble
- Use standard Markdown headings (`#`, `##`, `###`, `####`) reflecting document hierarchy (e.g. Report Title, Chapter, Section).
- NEVER use LaTeX environments like `\begin{center}` or `\begin{flushleft}` for plain text, titles, or forewords. Use native Markdown headings and paragraphs.
- For horizontal divider rules, ALWAYS use `---` (never ASCII underscores `____`).

### 2. Multi-Column Blocks & Header Metadata
- Multi-column horizontal headers (e.g. Volume number, Report status, Date) MUST be rendered as a clean Markdown table with headers.
- For multi-column bibliographic cards, catalog cards, or metadata boxes, reconstruct text in natural reading order (column-by-column, top-to-bottom). Do NOT interleave text across adjacent columns.
- Do NOT transcribe external library accession stamps, date-received ink stamps, or library barcodes.
- Carefully distinguish Roman numerals (e.g., 'ii', 'iii', 'iv') from Arabic numbers (do NOT transcribe 'ii' as '11').

### 3. Verbatim Content & Inline Typography
- Transcribe exact wording, capitalization, punctuation, and section codes (e.g., `**5.12.3.3 SEAL SEATING LOAD.**`).
- Preserve bold (`**text**`) and italic (`*text*`) styling exactly as shown in names, titles, publishers, and signature lines.
- Preserve HTML underline tags `<u>...</u>` where present in section lead-ins and technical definitions.
- Maintain natural paragraph flow: do NOT insert artificial hard line breaks mid-sentence.
- Preserve blank lines between distinct list entries, report volumes, or paragraphs.

### 4. LaTeX Equations & Mathematical Notation
- Inline variables, indices, and math symbols MUST be enclosed in `$ ... $` (e.g., `$D_{eff}$`, `$\tau$`, `$\sigma_m$`).
- Matrix and vector identifiers with brackets MUST use LaTeX with Roman font: e.g., `$[\mathrm{LB}]'$`, `$[\mathrm{IG}]$`, `$[\mathrm{GB}]$`.
- Display equations MUST use `$$ ... $$` and include equation tags where present: `$$ <equation> \tag{...} $$`.
- Align multi-line derivations or variable definitions using `\begin{aligned} ... \end{aligned}`.
- Format matrices with `\begin{bmatrix} ... \end{bmatrix}` with clean multi-line layout and proper `\\` breaks.

### 5. Strict Output Enforcement
- Output ONLY the verbatim Markdown text.
- Do NOT include markdown code block backticks (````markdown or ```) around the entire output.
- Do NOT include any thoughts, reasoning, preambles, notes, or conversational commentary."#.to_string()),
                        inline_data: None,
                    },
                    Part {
                        text: None,
                        inline_data: Some(InlineData {
                            mime_type: "image/png".to_string(),
                            data: image_base64,
                        }),
                    },
                ],
            }],
            generation_config: Some(GenerationConfig {
                temperature: Some(0.0),
            }),
        };

        let client = reqwest::Client::new();
        let endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent";



        let response = client
            .post(format!("{endpoint}?key={api_key}"))
            .json(&payload)
            .send()
            .await
            .context("Gemini image OCR request failed")?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .context("failed to read Gemini OCR response body")?;

        if !status.is_success() {
            bail!("Gemini image OCR returned HTTP {status}: {response_text}");
        }

        let parsed: GeminiResponse = serde_json::from_str(&response_text)
            .context("failed to parse Gemini OCR response JSON")?;

        let raw_text = parsed
            .candidates
            .into_iter()
            .flat_map(|candidate| candidate.content.into_iter().flat_map(|content| content.parts))
            .filter(|part| !part.thought.unwrap_or(false))
            .filter_map(|part| part.text)
            .collect::<Vec<_>>()
            .join("\n");

        let trimmed = raw_text.trim();
        let cleaned = if (trimmed.starts_with("```markdown") || trimmed.starts_with("```"))
            && trimmed.ends_with("```")
        {
            let lines: Vec<&str> = trimmed.lines().collect();
            if lines.len() >= 2 {
                lines[1..lines.len() - 1].join("\n")
            } else {
                trimmed.to_string()
            }
        } else {
            raw_text
        };

        Ok::<String, anyhow::Error>(cleaned)
    })
}
