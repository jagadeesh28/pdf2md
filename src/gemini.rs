use anyhow::{Context, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<Content>,
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
                        text: Some(r#"Transcribe all visible content from this PDF page into Obsidian-compatible Markdown.
Preserve the document's headings, paragraphs, lists, tables, emphasis, links, and reading order(sometimes make have two or more columns). Do not add commentary, explanations, or Markdown code fences. If there is no text, return an empty string.
Represent tables faithfully using Markdown syntax, including headers, rows, and columns.
Represent mathematics using LaTeX delimiters supported by Obsidian: use $...$ for inline equations, formulas, and mathematical symbols, and $$...$$ on separate lines for displayed or complex equations. Use standard LaTeX commands inside math delimiters (for example, \frac{a}{b}, \sum, \alpha, and \mathbb{R}) rather than replacing mathematical notation with prose. Keep equations faithful to the page and do not invent missing content."#.to_string()),
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
        };

        let client = reqwest::Client::new();
        let endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent";
        // "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent";
        //"https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent";

        // https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent
        // https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent


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

        let text = parsed
            .candidates
            .into_iter()
            .flat_map(|candidate| candidate.content.into_iter().flat_map(|content| content.parts))
            .filter_map(|part| part.text)
            .collect::<Vec<_>>()
            .join("\n");

        Ok::<String, anyhow::Error>(text)
    })
}
