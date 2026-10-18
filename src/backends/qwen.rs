use anyhow::{Context, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_last_n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
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
5. Mathematics: Represent all mathematical notation strictly using Obsidian Markdown LaTeX delimiters:
   - For inline formulas and variables: ALWAYS use single dollar delimiters $...$ (e.g. $T_s$, $u_g$, $x$, $\rho_p$). NEVER use \(...\).
   - For display/block equations: ALWAYS use double dollar delimiters ($$ ... $$) on separate lines. NEVER use \[...\].
6. Typography: Preserve exact wording, capitalization, punctuation, bold (**text**), italic (*text*), and Roman numerals (e.g., 'ii', 'iii', 'iv'). Do not transcribe accession stamps, ink date stamps, or barcodes.
7. Figures and Captions: For any figures, diagrams, plots, charts, or illustrations on the page, always transcribe the figure number and complete caption verbatim in the Markdown text at the location where they appear (e.g. `*Figure 4.5.—Caption text...*`). Do NOT omit figure numbers or figure captions from the text transcription."#;

const DEFAULT_USER_PROMPT: &str = "Transcribe all visible content from this document page verbatim into Obsidian-compatible Markdown.";

const DEFAULT_TEXT_SYSTEM_PROMPT: &str = r#"You are an expert technical document transcription engine.
Your task is to reformat and transcribe the provided raw text from a document page into verbatim, publication-quality Obsidian-compatible Markdown.

Adhere strictly to these formatting rules:
1. Output ONLY the verbatim Markdown document. Never include preambles, explanations, thinking steps, or markdown code block fences (e.g. ```markdown) around the output.
2. Structure: Use standard Markdown headings (#, ##, ###, ####) reflecting document hierarchy.
3. Tables: Represent tables faithfully using Markdown table syntax.
4. Mathematics: Represent all mathematical formulas and variables strictly using Obsidian Markdown LaTeX delimiters:
   - For inline formulas and variables: ALWAYS use single dollar delimiters $...$ (e.g. $T_s$, $u_g$, $x$, $\rho_p$). NEVER use \(...\).
   - For display/block equations: ALWAYS use double dollar delimiters ($$ ... $$) on separate lines. NEVER use \[...\].
5. Clean up OCR/extraction artifacts and fix line breaks mid-sentence to restore natural reading flow.
6. Figures and Captions: Always preserve and transcribe all figure numbers and figure captions verbatim in the Markdown text."#;

/// Resolves the model to use by querying Ollama `/api/tags` or environment variables.
pub fn resolve_model_name(client: &reqwest::blocking::Client, base_url: &str) -> String {
    if let Ok(model) = std::env::var("QWEN_MODEL")
        && !model.trim().is_empty()
    {
        return model;
    }

    // Try detecting installed models from Ollama /api/tags
    let tags_url = if let Ok(mut parsed) = reqwest::Url::parse(base_url) {
        parsed.set_path("/api/tags");
        parsed.to_string()
    } else {
        "http://localhost:11434/api/tags".to_string()
    };

    if let Ok(resp) = client.get(&tags_url).send()
        && let Ok(body) = resp.text()
        && let Ok(parsed) = serde_json::from_str::<OllamaTagsResponse>(&body)
    {
        // If a multimodal / vision model is present, prefer it for OCR
        if let Some(m) = parsed.models.iter().find(|m| {
            let name = m.name.to_lowercase();
            name.contains("vl") || name.contains("vision") || name.contains("llava")
        }) {
            return m.name.clone();
        }
        // Otherwise find any Qwen model (e.g. qwen2.5-coder:32b-instruct-q4_K_M)
        if let Some(m) = parsed
            .models
            .iter()
            .find(|m| m.name.to_lowercase().contains("qwen"))
        {
            return m.name.clone();
        }
        if let Some(m) = parsed.models.first() {
            return m.name.clone();
        }
    }

    "qwen2.5-coder:32b-instruct-q4_K_M".to_string()
}

pub fn is_ollama_endpoint(url: &str) -> bool {
    url.contains("11434") || url.ends_with("/api/chat")
}

pub fn normalize_ollama_chat_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/api/chat") {
        trimmed.to_string()
    } else if trimmed.ends_with("/v1") {
        let prefix = trimmed.trim_end_matches("/v1");
        format!("{prefix}/api/chat")
    } else {
        format!("{trimmed}/api/chat")
    }
}

pub fn normalize_chat_completions_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else if trimmed.ends_with("/v1") {
        format!("{trimmed}/chat/completions")
    } else {
        format!("{trimmed}/v1/chat/completions")
    }
}

pub fn create_http_client() -> reqwest::blocking::Client {
    let timeout_secs = std::env::var("PDF2MD_TIMEOUT_SECS")
        .or_else(|_| std::env::var("QWEN_TIMEOUT_SECS"))
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(600);

    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new())
}

pub fn ocr_image_file(image_path: &Path) -> Result<String> {
    let bytes = fs::read(image_path).context("failed to read OCR image file")?;
    let image_base64 = STANDARD.encode(&bytes);

    let base_url = std::env::var("QWEN_URL")
        .or_else(|_| std::env::var("QWEN_API_BASE"))
        .or_else(|_| std::env::var("LOCAL_LLM_URL"))
        .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());

    let client = create_http_client();
    let model = resolve_model_name(&client, &base_url);
    let temperature: f32 = std::env::var("QWEN_TEMPERATURE")
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or(0.1);

    let repeat_penalty: f32 = std::env::var("QWEN_REPEAT_PENALTY")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(1.15);

    let presence_penalty: f32 = std::env::var("QWEN_PRESENCE_PENALTY")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(0.1);

    let frequency_penalty: f32 = std::env::var("QWEN_FREQUENCY_PENALTY")
        .ok()
        .and_then(|f| f.parse().ok())
        .unwrap_or(0.1);

    let api_key = std::env::var("QWEN_API_KEY").unwrap_or_default();

    let stop_tokens = vec!["<|im_end|>".to_string(), "<|endoftext|>".to_string()];

    let is_ollama = is_ollama_endpoint(&base_url);
    let ollama_endpoint = normalize_ollama_chat_url(&base_url);
    let openai_endpoint = normalize_chat_completions_url(&base_url);

    let max_retries = 2;
    let mut cur_repeat_penalty = repeat_penalty;
    let mut cur_temperature = temperature;
    let mut cur_repeat_last_n = 64;
    let mut cur_presence_penalty = presence_penalty;
    let mut cur_frequency_penalty = frequency_penalty;

    for attempt in 0..=max_retries {
        let (status, body) = if is_ollama {
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
                        images: vec![image_base64.clone()],
                    },
                ],
                stream: false,
                options: Some(OllamaOptions {
                    temperature: Some(cur_temperature),
                    repeat_penalty: Some(cur_repeat_penalty),
                    repeat_last_n: Some(cur_repeat_last_n),
                    stop: Some(stop_tokens.clone()),
                }),
            };

            let mut request = client.post(&ollama_endpoint).json(&payload);
            if !api_key.is_empty() {
                request = request.header("Authorization", format!("Bearer {api_key}"));
            }

            let response = request
                .send()
                .with_context(|| format!("failed to send request to local Qwen endpoint {ollama_endpoint} (Make sure your local server is running, e.g. 'ollama serve')"))?;

            let status = response.status();
            let body = response
                .text()
                .context("failed to read Qwen response body")?;
            (status, body)
        } else {
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
                temperature: Some(cur_temperature),
                presence_penalty: Some(cur_presence_penalty),
                frequency_penalty: Some(cur_frequency_penalty),
                stop: Some(stop_tokens.clone()),
            };

            let mut request = client.post(&openai_endpoint).json(&payload);
            if !api_key.is_empty() {
                request = request.header("Authorization", format!("Bearer {api_key}"));
            }

            let response = request
                .send()
                .with_context(|| format!("failed to send request to local Qwen endpoint {openai_endpoint} (Make sure your local server is running, e.g. 'ollama serve' or vLLM)"))?;

            let status = response.status();
            let body = response
                .text()
                .context("failed to read Qwen response body")?;
            (status, body)
        };

        if !status.is_success() {
            if body.contains("token repeat limit reached") && attempt < max_retries {
                cur_repeat_penalty = if attempt == 0 {
                    cur_repeat_penalty.max(1.3)
                } else {
                    cur_repeat_penalty.max(1.5)
                };
                cur_temperature = (cur_temperature + 0.2).min(0.7);
                cur_repeat_last_n = if attempt == 0 { 128 } else { 256 };
                cur_presence_penalty = (cur_presence_penalty + 0.3).min(1.5);
                cur_frequency_penalty = (cur_frequency_penalty + 0.3).min(1.5);
                eprintln!(
                    "Notice: Local model '{model}' entered a repetitive token loop (attempt {}/{}). Retrying with repeat_penalty={:.2}, temperature={:.2}...",
                    attempt + 1,
                    max_retries + 1,
                    cur_repeat_penalty,
                    cur_temperature
                );
                continue;
            }
            handle_api_error(status, &body, &model)?;
        }

        let parsed: UnifiedChatResponse = serde_json::from_str(&body)
            .with_context(|| format!("failed to parse Qwen JSON response: {body}"))?;

        let raw_text = if is_ollama {
            parsed
                .message
                .and_then(|m| m.content)
                .or(parsed.response)
                .unwrap_or_default()
        } else {
            extract_content_from_response(&parsed).unwrap_or_else(|| body.clone())
        };

        return Ok(clean_model_markdown(&raw_text));
    }

    bail!(
        "Local model '{model}' entered a repetitive token loop and could not recover after {} retries.",
        max_retries
    )
}

pub fn transcribe_text(text: &str) -> Result<String> {
    let base_url = std::env::var("QWEN_URL")
        .or_else(|_| std::env::var("QWEN_API_BASE"))
        .or_else(|_| std::env::var("LOCAL_LLM_URL"))
        .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());

    let client = create_http_client();
    let model = resolve_model_name(&client, &base_url);
    let api_key = std::env::var("QWEN_API_KEY").unwrap_or_default();

    let is_ollama = is_ollama_endpoint(&base_url);
    let ollama_endpoint = normalize_ollama_chat_url(&base_url);
    let openai_endpoint = normalize_chat_completions_url(&base_url);

    let (status, body) = if is_ollama {
        let payload = OllamaChatRequest {
            model: model.clone(),
            messages: vec![
                OllamaMessage {
                    role: "system".to_string(),
                    content: DEFAULT_TEXT_SYSTEM_PROMPT.to_string(),
                    images: vec![],
                },
                OllamaMessage {
                    role: "user".to_string(),
                    content: format!(
                        "Reformat and transcribe the following document text into Obsidian Markdown:\n\n{text}"
                    ),
                    images: vec![],
                },
            ],
            stream: false,
            options: Some(OllamaOptions {
                temperature: Some(0.1),
                repeat_penalty: Some(1.2),
                repeat_last_n: Some(64),
                stop: None,
            }),
        };

        let mut request = client.post(&ollama_endpoint).json(&payload);
        if !api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request.send().with_context(|| {
            format!("failed to send text request to Qwen endpoint {ollama_endpoint}")
        })?;

        let status = response.status();
        let body = response.text().context("failed to read response body")?;
        (status, body)
    } else {
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
            temperature: Some(0.1),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stop: None,
        };

        let mut request = client.post(&openai_endpoint).json(&payload);
        if !api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request.send().with_context(|| {
            format!("failed to send text request to Qwen endpoint {openai_endpoint}")
        })?;

        let status = response.status();
        let body = response.text().context("failed to read response body")?;
        (status, body)
    };

    if !status.is_success() {
        handle_api_error(status, &body, &model)?;
    }

    let parsed: UnifiedChatResponse = serde_json::from_str(&body)
        .with_context(|| format!("failed to parse Qwen JSON response: {body}"))?;

    let raw = extract_content_from_response(&parsed).unwrap_or(body);
    Ok(clean_model_markdown(&raw))
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
    if body.contains("token repeat limit reached") {
        bail!(
            "Local model '{model}' entered a repetitive token loop and was stopped by Ollama (token repeat limit reached).\n\
             This can happen on complex pages with repeated borders or dense symbols.\n\
             Try adjusting repeat penalty (e.g. export QWEN_REPEAT_PENALTY=1.2) or use Gemini/Gemma for this document:\n  \
             cargo run --release -- <input.pdf> <output.md> <start> <end> gemini"
        );
    }
    bail!("Local Qwen endpoint returned HTTP {status}: {body}");
}

fn extract_content_from_response(parsed: &UnifiedChatResponse) -> Option<String> {
    if let Some(choice) = parsed.choices.first() {
        if let Some(message) = &choice.message
            && let Some(content) = &message.content
        {
            return Some(content.clone());
        }
        if let Some(text) = &choice.text {
            return Some(text.clone());
        }
    }
    if let Some(message) = &parsed.message
        && let Some(content) = &message.content
    {
        return Some(content.clone());
    }
    parsed.response.clone()
}

pub fn clean_model_markdown(raw_text: &str) -> String {
    let mut text = raw_text.to_string();

    // Strip <think>...</think> if reasoning thoughts are present
    if let Some(start) = text.find("<think>")
        && let Some(end) = text.find("</think>")
    {
        let after_end = end + "</think>".len();
        text = format!("{}{}", &text[..start], &text[after_end..]);
    }

    let trimmed = text.trim();
    let stripped = if (trimmed.starts_with("```markdown") || trimmed.starts_with("```"))
        && trimmed.ends_with("```")
    {
        let lines: Vec<&str> = trimmed.lines().collect();
        if lines.len() >= 2 {
            lines[1..lines.len() - 1].join("\n")
        } else {
            trimmed.to_string()
        }
    } else {
        trimmed.to_string()
    };

    crate::markdown::normalize_latex_delimiters(&stripped)
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
    fn test_ollama_endpoint_detection_and_url() {
        assert!(is_ollama_endpoint("http://localhost:11434"));
        assert!(is_ollama_endpoint("http://localhost:11434/v1"));
        assert!(is_ollama_endpoint("http://127.0.0.1:11434/api/chat"));
        assert!(!is_ollama_endpoint("http://localhost:8000/v1"));

        assert_eq!(
            normalize_ollama_chat_url("http://localhost:11434"),
            "http://localhost:11434/api/chat"
        );
        assert_eq!(
            normalize_ollama_chat_url("http://localhost:11434/v1"),
            "http://localhost:11434/api/chat"
        );
        assert_eq!(
            normalize_ollama_chat_url("http://localhost:11434/api/chat"),
            "http://localhost:11434/api/chat"
        );
    }

    #[test]
    fn test_clean_model_markdown() {
        let code_fenced = "```markdown\n# Title\nSome content\n```";
        assert_eq!(clean_model_markdown(code_fenced), "# Title\nSome content");

        let with_think = "<think>\nThinking about the image...\n</think>\n# Output";
        assert_eq!(clean_model_markdown(with_think), "# Output");

        let with_latex = "Pressure \\( p \\) and equation:\n\\[\nE = mc^2\n\\]";
        assert_eq!(
            clean_model_markdown(with_latex),
            "Pressure $p$ and equation:\n$$\nE = mc^2\n$$"
        );
    }
}
