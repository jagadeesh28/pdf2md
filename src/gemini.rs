use anyhow::{Context, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize)]
struct GeminiRequest {
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    contents: Vec<Content>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Debug, Serialize)]
struct GenerationConfig {
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    thinking_config: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct ThinkingConfig {
    #[serde(rename = "thinkingLevel", skip_serializing_if = "Option::is_none")]
    thinking_level: Option<String>,
    #[serde(rename = "thinkingBudget", skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<i32>,
}

#[derive(Debug, Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
struct Part {
    #[serde(rename = "text", skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(rename = "inlineData", skip_serializing_if = "Option::is_none")]
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
        let system_prompt = r#"You are an expert OCR and document transcription engine specializing in technical and scientific reports.
Your task is to transcribe document page images into verbatim, publication-quality Obsidian-compatible Markdown.

Adhere strictly to these formatting rules:
1. Transcribe ONLY the visible document content. Never include preambles, conversational notes, thinking steps, image descriptions, or markdown code block fences (e.g. ```markdown) around the output.
2. Structure: Use standard Markdown headings (#, ##, ###, ####) reflecting document hierarchy.
3. Multi-Column: For multi-column pages or bibliographic metadata blocks, transcribe in natural reading order (column-by-column, top-to-bottom). Do not interleave text across adjacent columns.
4. Tables: Represent tables faithfully using Markdown table syntax.
5. Mathematics: Represent all mathematical notation using LaTeX delimiters supported by Obsidian: use $...$ for inline formulas/variables, and $$...$$ on separate lines for display equations.
6. Typography: Preserve exact wording, capitalization, punctuation, bold (**text**), italic (*text*), and Roman numerals (e.g., 'ii', 'iii', 'iv'). Do not transcribe accession stamps, ink date stamps, or barcodes."#;

        let user_prompt = "Transcribe all visible content from this document page verbatim into Obsidian-compatible Markdown.";

        let model = std::env::var("GEMINI_MODEL")
            .unwrap_or_else(|_| "gemini-3.5-flash-lite".to_string());

        let thinking_config = if model.starts_with("gemini-2.5") {
            Some(ThinkingConfig {
                thinking_level: None,
                thinking_budget: Some(0),
            })
        } else if model.starts_with("gemma") {
            // Gemma models do not support thinkingConfig (returns HTTP 400)
            None
        } else {
            Some(ThinkingConfig {
                thinking_level: Some("low".to_string()),
                thinking_budget: None,
            })
        };

        let payload = GeminiRequest {
            system_instruction: Some(Content {
                parts: vec![Part {
                    text: Some(system_prompt.to_string()),
                    inline_data: None,
                }],
            }),
            contents: vec![Content {
                parts: vec![
                    Part {
                        text: Some(user_prompt.to_string()),
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
                thinking_config,
                temperature: Some(0.0),
            }),
        };

        let client = reqwest::Client::new();
        let endpoint = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        );

        let response = client
            .post(endpoint)
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
