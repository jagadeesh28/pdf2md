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
                        text: Some(r#"[SYSTEM: CRITICAL INSTRUCTION]
You operate exclusively as a literal OCR transcription tool. Do not think out loud. Do not create "Transcription Plans", "Image Analysis", or "Detailed Transcriptions". 

Your final answer must contain ONLY the raw markdown content inside the <transcription> tags below. Everything else must be empty.

<instructions>
1. Output the verbatim text exactly as it appears in the image.
2. Preserve reading order, headings, lists, tables, and typography.
3. Represent math using Obsidian LaTeX delimiters: ... for inline, \[...\] on new lines for block equations.
4. ZERO CHATTER: Do not output markdown code blocks (```), do not say "Here is your transcription", and do not write an introduction or conclusion.
</instructions>

<transcription>
"#.to_string()),
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
