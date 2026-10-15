pub mod gemini;
pub mod gemma;
pub mod qwen;

use anyhow::{Result, anyhow};
use std::fmt;
use std::str::FromStr;

/// Supported OCR and transcription backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OcrBackend {
    /// Locally deployed Qwen model (via Ollama or OpenAI-compatible endpoint).
    #[default]
    Qwen,
    /// Google Gemini cloud API.
    Gemini,
    /// Google Gemma cloud API.
    Gemma,
}

impl FromStr for OcrBackend {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "qwen" | "qwen-coder" | "qwencoder" | "local" => Ok(OcrBackend::Qwen),
            "gemini" => Ok(OcrBackend::Gemini),
            "gemma" => Ok(OcrBackend::Gemma),
            other => Err(anyhow!(
                "unknown OCR backend '{other}'; valid options are: qwen, gemini, gemma"
            )),
        }
    }
}

impl fmt::Display for OcrBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OcrBackend::Qwen => write!(f, "qwen"),
            OcrBackend::Gemini => write!(f, "gemini"),
            OcrBackend::Gemma => write!(f, "gemma"),
        }
    }
}

/// Resolves the default backend from `PDF2MD_BACKEND` or `OCR_BACKEND` env vars, defaulting to Qwen.
pub fn resolve_default_backend() -> OcrBackend {
    if let Ok(val) = std::env::var("PDF2MD_BACKEND").or_else(|_| std::env::var("OCR_BACKEND")) {
        val.parse().unwrap_or(OcrBackend::Qwen)
    } else {
        OcrBackend::Qwen
    }
}
