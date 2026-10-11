use anyhow::{Context, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: String,
    content: MessageContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlObject },
}

#[derive(Debug, Serialize)]
struct ImageUrlObject {
    url: String,
}

#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Debug, Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    images: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    temperature: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct UnifiedChatResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    message: Option<ChatMessage>,
    #[serde(default)]
    response: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: Option<ChatMessage>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    #[serde(default)]
    models: Vec<OllamaModelTag>,
}

#[derive(Debug, Deserialize)]
struct OllamaModelTag {
    name: String,
}

const DEFAULT_SYSTEM_PROMPT: &str = r#"You are an expert OCR and document transcription engine specializing in technical and scientific reports.
Your task is to transcribe document page images into verbatim, publication-quality Obsidian-compatible Markdown.

Adhere strictly to these formatting rules:
1. Transcribe ONLY the visible document content. Never include preambles, conversational notes, thinking steps, image descriptions, or markdown code block fences (e.g. ```markdown) around the output.
2. Structure: Use standard Markdown headings (#, ##, ###, ####) reflecting document hierarchy.
3. Multi-Column: For multi-column pages or bibliographic metadata blocks, transcribe in natural reading order (column-by-column, top-to-bottom). Do not interleave text across adjacent columns.
4. Tables: Represent tables faithfully using Markdown table syntax.
5. Mathematics: Represent all mathematical notation using LaTeX delimiters supported by Obsidian: use $...$ for inline formulas/variables, and $$...$$ on separate lines for display equations.
6. Typography: Preserve exact wording, capitalization, punctuation, bold (**text**), italic (*text*), and Roman numerals (e.g., 'ii', 'iii', 'iv'). Do not transcribe accession stamps, ink date stamps, or barcodes."#;

const DEFAULT_USER_PROMPT: &str =
    "Transcribe all visible content from this document page verbatim into Obsidian-compatible Markdown.";

const DEFAULT_TEXT_SYSTEM_PROMPT: &str = r#"You are an expert technical document transcription engine.
Your task is to reformat and transcribe the provided raw text from a document page into verbatim, publication-quality Obsidian-compatible Markdown.

Adhere strictly to these formatting rules:
1. Output ONLY the verbatim Markdown document. Never include preambles, explanations, thinking steps, or markdown code block fences (e.g. ```markdown) around the output.
2. Structure: Use standard Markdown headings (#, ##, ###, ####) reflecting document hierarchy.
3. Tables: Represent tables faithfully using Markdown table syntax.
4. Mathematics: Represent mathematical formulas and variables using LaTeX delimiters supported by Obsidian ($...$ for inline, $$...$$ for block).
5. Clean up OCR/extraction artifacts and fix line breaks mid-sentence to restore natural reading flow."#;

pub async fn resolve_model_name(client: &reqwest::Client, base_url: &str) -> String {
    if let Ok(model) = std::env::var("QWEN_MODEL") {
        if !model.trim().is_empty() {
            return model;
        }
    }

    // Try detecting installed models from Ollama /api/tags
    let tags_url = if let Ok(mut parsed) = reqwest::Url::parse(base_url) {
        parsed.set_path("/api/tags");
        parsed.to_string()
    } else {
        "http://localhost:11434/api/tags".to_string()
    };

    if let Ok(resp) = client.get(&tags_url).send().await {
        if let Ok(body) = resp.text().await {
            if let Ok(parsed) = serde_json::from_str::<OllamaTagsResponse>(&body) {
                // If a multimodal / vision model is present, prefer it for OCR
                if let Some(m) = parsed.models.iter().find(|m| {
                    let name = m.name.to_lowercase();
                    name.contains("vl") || name.contains("vision") || name.contains("llava")
                }) {
                    return m.name.clone();
                }
                // Otherwise find any Qwen model (e.g. qwen2.5-coder:32b-instruct-q4_K_M)
                if let Some(m) = parsed.models.iter().find(|m| m.name.to_lowercase().contains("qwen")) {
                    return m.name.clone();
                }
                if let Some(m) = parsed.models.first() {
                    return m.name.clone();
                }
            }
        }
    }

    "qwen2.5-coder:32b-instruct-q4_K_M".to_string()
}

pub fn ocr_image_file(image_path: &Path) -> Result<String> {
    let bytes = fs::read(image_path).context("failed to read OCR image file")?;
    let image_base64 = STANDARD.encode(&bytes);

    let runtime = tokio::runtime::Runtime::new()
        .context("failed to create Tokio runtime for local Qwen OCR")?;

    runtime.block_on(async {
        let base_url = std::env::var("QWEN_URL")
            .or_else(|_| std::env::var("QWEN_API_BASE"))
            .or_else(|_| std::env::var("LOCAL_LLM_URL"))
            .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());

        let client = reqwest::Client::new();
        let model = resolve_model_name(&client, &base_url).await;
        let api_key = std::env::var("QWEN_API_KEY").unwrap_or_default();

        let raw_text = if base_url.ends_with("/api/chat") {
            // Native Ollama chat endpoint
            let payload = OllamaChatRequest {
                model: model.clone(),
                messages: vec![
                    OllamaMessage {
                        role: "system".to_string(),
                        content: DEFAULT_SYSTEM_PROMPT.to_string(),
                        images: vec![],
                    },
                    OllamaMessage {
                        role: "user".to_string(),
                        content: DEFAULT_USER_PROMPT.to_string(),
                        images: vec![image_base64],
                    },
                ],
                stream: false,
                options: Some(OllamaOptions {
                    temperature: Some(0.0),
                }),
            };

            let mut request = client.post(&base_url).json(&payload);
            if !api_key.is_empty() {
                request = request.header("Authorization", format!("Bearer {api_key}"));
            }

            let response = request
                .send()
                .await
                .with_context(|| format!("failed to send request to local Qwen endpoint {base_url} (Make sure your local server is running, e.g. 'ollama serve')"))?;

            let status = response.status();
            let body = response
                .text()
                .await
                .context("failed to read Qwen response body")?;

            if !status.is_success() {
                handle_api_error(status, &body, &model)?;
            }

            let parsed: UnifiedChatResponse = serde_json::from_str(&body)
                .with_context(|| format!("failed to parse Qwen JSON response: {body}"))?;

            parsed
                .message
                .and_then(|m| m.content)
                .or(parsed.response)
                .unwrap_or_default()
        } else {
            // OpenAI-compatible endpoint (Ollama /v1/chat/completions, vLLM, LMDeploy, etc.)
            let endpoint = normalize_chat_completions_url(&base_url);
            let payload = OpenAiChatRequest {
                model: model.clone(),
                messages: vec![
                    OpenAiMessage {
                        role: "system".to_string(),
                        content: MessageContent::Text(DEFAULT_SYSTEM_PROMPT.to_string()),
                    },
                    OpenAiMessage {
                        role: "user".to_string(),
                        content: MessageContent::Parts(vec![
                            ContentPart::Text {
                                text: DEFAULT_USER_PROMPT.to_string(),
                            },
                            ContentPart::ImageUrl {
                                image_url: ImageUrlObject {
                                    url: format!("data:image/png;base64,{image_base64}"),
                                },
                            },
                        ]),
                    },
                ],
                temperature: Some(0.0),
            };

            let mut request = client.post(&endpoint).json(&payload);
            if !api_key.is_empty() {
                request = request.header("Authorization", format!("Bearer {api_key}"));
            }

            let response = request
                .send()
                .await
                .with_context(|| format!("failed to send request to local Qwen endpoint {endpoint} (Make sure your local server is running, e.g. 'ollama serve' or vLLM)"))?;

            let status = response.status();
            let body = response
                .text()
                .await
                .context("failed to read Qwen response body")?;

            if !status.is_success() {
                handle_api_error(status, &body, &model)?;
            }

            let parsed: UnifiedChatResponse = serde_json::from_str(&body)
                .with_context(|| format!("failed to parse Qwen JSON response: {body}"))?;

            extract_content_from_response(&parsed)
                .unwrap_or_else(|| body.clone())
        };

        Ok(clean_model_markdown(&raw_text))
    })
}

pub fn transcribe_text(text: &str) -> Result<String> {
    let runtime = tokio::runtime::Runtime::new()
        .context("failed to create Tokio runtime for local Qwen text transcription")?;

    runtime.block_on(async {
        let base_url = std::env::var("QWEN_URL")
            .or_else(|_| std::env::var("QWEN_API_BASE"))
            .or_else(|_| std::env::var("LOCAL_LLM_URL"))
            .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());

        let client = reqwest::Client::new();
        let model = resolve_model_name(&client, &base_url).await;
        let api_key = std::env::var("QWEN_API_KEY").unwrap_or_default();

        let endpoint = normalize_chat_completions_url(&base_url);
        let payload = OpenAiChatRequest {
            model: model.clone(),
            messages: vec![
                OpenAiMessage {
                    role: "system".to_string(),
                    content: MessageContent::Text(DEFAULT_TEXT_SYSTEM_PROMPT.to_string()),
                },
                OpenAiMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text(format!(
                        "Reformat and transcribe the following document text into Obsidian Markdown:\n\n{text}"
                    )),
                },
            ],
            temperature: Some(0.0),
        };

        let mut request = client.post(&endpoint).json(&payload);
        if !api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request
            .send()
            .await
            .with_context(|| format!("failed to send text request to Qwen endpoint {endpoint}"))?;

        let status = response.status();
        let body = response.text().await.context("failed to read response body")?;

        if !status.is_success() {
            handle_api_error(status, &body, &model)?;
        }

        let parsed: UnifiedChatResponse = serde_json::from_str(&body)
            .with_context(|| format!("failed to parse Qwen JSON response: {body}"))?;

        let raw = extract_content_from_response(&parsed).unwrap_or(body);
        Ok(clean_model_markdown(&raw))
    })
}

fn handle_api_error(status: reqwest::StatusCode, body: &str, model: &str) -> Result<()> {
    if body.contains("does not support multimodal") {
        bail!(
            "Local model '{model}' does not support multimodal (vision) requests.\n\
             To transcribe document images directly with a local model, pull a vision model in Ollama:\n  \
             ollama pull qwen2.5-vl\n\
             export QWEN_MODEL=qwen2.5-vl\n\
             Or convert using Gemini / Gemma:\n  \
             cargo run -- <input.pdf> <output.md> <start> <end> gemini"
        );
    }
    if body.contains("not found") {
        bail!(
            "Model '{model}' was not found in Ollama.\n\
             Run 'ollama list' to check installed models, or set QWEN_MODEL (e.g. export QWEN_MODEL=qwen2.5-coder:32b-instruct-q4_K_M)."
        );
    }
    bail!("Local Qwen endpoint returned HTTP {status}: {body}");
}

fn normalize_chat_completions_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else if trimmed.ends_with("/v1") {
        format!("{trimmed}/chat/completions")
    } else {
        format!("{trimmed}/v1/chat/completions")
    }
}

fn extract_content_from_response(parsed: &UnifiedChatResponse) -> Option<String> {
    if let Some(choice) = parsed.choices.first() {
        if let Some(message) = &choice.message {
            if let Some(content) = &message.content {
                return Some(content.clone());
            }
        }
        if let Some(text) = &choice.text {
            return Some(text.clone());
        }
    }
    if let Some(message) = &parsed.message {
        if let Some(content) = &message.content {
            return Some(content.clone());
        }
    }
    parsed.response.clone()
}

pub fn clean_model_markdown(raw_text: &str) -> String {
    let mut text = raw_text.to_string();

    // Strip <think>...</think> if reasoning thoughts are present
    if let Some(start) = text.find("<think>") {
        if let Some(end) = text.find("</think>") {
            let after_end = end + "</think>".len();
            text = format!("{}{}", &text[..start], &text[after_end..]);
        }
    }

    let trimmed = text.trim();
    if (trimmed.starts_with("```markdown") || trimmed.starts_with("```")) && trimmed.ends_with("```")
    {
        let lines: Vec<&str> = trimmed.lines().collect();
        if lines.len() >= 2 {
            lines[1..lines.len() - 1].join("\n")
        } else {
            trimmed.to_string()
        }
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_chat_completions_url() {
        assert_eq!(
            normalize_chat_completions_url("http://localhost:11434"),
            "http://localhost:11434/v1/chat/completions"
        );
        assert_eq!(
            normalize_chat_completions_url("http://localhost:11434/v1"),
            "http://localhost:11434/v1/chat/completions"
        );
        assert_eq!(
            normalize_chat_completions_url("http://localhost:8000/v1/chat/completions/"),
            "http://localhost:8000/v1/chat/completions"
        );
    }

    #[test]
    fn test_clean_model_markdown() {
        let code_fenced = "```markdown\n# Title\nSome content\n```";
        assert_eq!(clean_model_markdown(code_fenced), "# Title\nSome content");

        let with_think = "<think>\nThinking about the image...\n</think>\n# Output";
        assert_eq!(clean_model_markdown(with_think), "# Output");
    }
}
