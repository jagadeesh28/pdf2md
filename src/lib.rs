pub mod fallback;
pub mod gemini;
pub mod gemma;
pub mod qwen;

use anyhow::{Context, Result, anyhow};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OcrBackend {
    #[default]
    Qwen,
    Gemini,
    Gemma,
}

impl std::str::FromStr for OcrBackend {
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

impl std::fmt::Display for OcrBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OcrBackend::Qwen => write!(f, "qwen"),
            OcrBackend::Gemini => write!(f, "gemini"),
            OcrBackend::Gemma => write!(f, "gemma"),
        }
    }
}

pub fn resolve_default_backend() -> OcrBackend {
    if let Ok(val) = std::env::var("PDF2MD_BACKEND").or_else(|_| std::env::var("OCR_BACKEND")) {
        val.parse().unwrap_or(OcrBackend::Qwen)
    } else {
        OcrBackend::Qwen
    }
}

#[derive(Debug, Default)]
pub struct PdfExtractionResult {
    pub text: String,
    pub images: Vec<PathBuf>,
}

#[derive(Debug, Default)]
pub struct ConversionConfig {
    pub temp_dir: Option<PathBuf>,
    pub output_images_dir: Option<PathBuf>,
}

pub fn process_pdf(
    input_path: &Path,
    output_path: &Path,
    begin_page: usize,
    end_page: usize,
) -> Result<()> {
    let backend = resolve_default_backend();
    process_pdf_with_backend(input_path, output_path, begin_page, end_page, backend)
}

pub fn process_pdf_with_backend(
    input_path: &Path,
    output_path: &Path,
    begin_page: usize,
    end_page: usize,
    backend: OcrBackend,
) -> Result<()> {
    validate_page_range(begin_page, end_page)?;

    let image_output_dir = output_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("images");
    let output_dir = output_path.parent().unwrap_or_else(|| Path::new("."));

    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create output directory {}", output_dir.display()))?;

    let mut extracted = PdfExtractionResult::default();
    process_ocr_pages_with_backend(
        input_path,
        &image_output_dir,
        begin_page,
        end_page,
        backend,
        |text, image| {
            if !text.trim().is_empty() {
                extracted.text.push_str(&text);
                if !extracted.text.ends_with("\n\n") {
                    extracted.text.push_str("\n\n");
                }
            }
            extracted.images.push(image);

            fs::write(output_path, render_markdown(&extracted)).with_context(|| {
                format!(
                    "failed to write markdown output to {}",
                    output_path.display()
                )
            })
        },
    )
    .with_context(|| format!("{backend} OCR failed for {}", input_path.display()))?;

    println!(
        "Converted {} to {} using {backend} backend",
        input_path.display(),
        output_path.display()
    );
    Ok(())
}

pub fn extract_pdf_text(
    input_path: &Path,
    begin_page: usize,
    end_page: usize,
) -> Result<PdfExtractionResult> {
    validate_page_range(begin_page, end_page)?;

    let bytes = fs::read(input_path)
        .with_context(|| format!("failed to read PDF file {}", input_path.display()))?;

    let pages = pdf_extract::extract_text_from_mem_by_pages(&bytes)
        .with_context(|| format!("failed to parse PDF {}", input_path.display()))?;
    if end_page > pages.len() {
        return Err(anyhow!(
            "page range {}-{} is outside the PDF's {} pages",
            begin_page,
            end_page,
            pages.len()
        ));
    }

    let text = pages[begin_page - 1..end_page].join("\n\n");

    let cleaned = clean_extracted_text(&text);
    if should_use_ocr_fallback(&cleaned) {
        return Err(anyhow::anyhow!(
            "No readable text was found in the PDF; the scanned-PDF OCR fallback is required."
        ));
    }

    Ok(PdfExtractionResult {
        text: cleaned,
        images: Vec::new(),
    })
}

pub fn fall_back_to_ocr(
    input_path: &Path,
    image_output_dir: &Path,
    begin_page: usize,
    end_page: usize,
) -> Result<PdfExtractionResult> {
    fall_back_to_ocr_with_backend(
        input_path,
        image_output_dir,
        begin_page,
        end_page,
        resolve_default_backend(),
    )
}

pub fn fall_back_to_ocr_with_backend(
    input_path: &Path,
    image_output_dir: &Path,
    begin_page: usize,
    end_page: usize,
    backend: OcrBackend,
) -> Result<PdfExtractionResult> {
    let mut result = PdfExtractionResult::default();
    process_ocr_pages_with_backend(
        input_path,
        image_output_dir,
        begin_page,
        end_page,
        backend,
        |text, image| {
            if !text.trim().is_empty() {
                result.text.push_str(&text);
                result.text.push_str("\n\n");
            }
            result.images.push(image);
            Ok(())
        },
    )?;

    Ok(result)
}

pub fn process_ocr_pages<F>(
    input_path: &Path,
    image_output_dir: &Path,
    begin_page: usize,
    end_page: usize,
    on_page: F,
) -> Result<()>
where
    F: FnMut(String, PathBuf) -> Result<()>,
{
    process_ocr_pages_with_backend(
        input_path,
        image_output_dir,
        begin_page,
        end_page,
        resolve_default_backend(),
        on_page,
    )
}

pub fn process_ocr_pages_with_backend<F>(
    input_path: &Path,
    image_output_dir: &Path,
    begin_page: usize,
    end_page: usize,
    backend: OcrBackend,
    mut on_page: F,
) -> Result<()>
where
    F: FnMut(String, PathBuf) -> Result<()>,
{
    validate_page_range(begin_page, end_page)?;

    let temp_dir = tempfile::tempdir().context("failed to create temporary OCR directory")?;
    let page_images =
        fallback::render_pages_to_images(input_path, temp_dir.path(), begin_page, end_page)
            .with_context(|| format!("failed to rasterize PDF {} for OCR", input_path.display()))?;

    fs::create_dir_all(image_output_dir).with_context(|| {
        format!(
            "failed to create image output directory {}",
            image_output_dir.display()
        )
    })?;

    for image_path in page_images {
        let ocr_text = match backend {
            OcrBackend::Qwen => qwen::ocr_image_file(&image_path)
                .with_context(|| format!("Qwen OCR failed for {}", image_path.display()))?,
            OcrBackend::Gemini => {
                let api_key = std::env::var("GEMINI_API_KEY").map_err(|_| {
                    anyhow!(
                        "Gemini OCR requires GEMINI_API_KEY. Set it before converting a PDF, e.g.: export GEMINI_API_KEY='your_api_key'"
                    )
                })?;
                gemini::ocr_image_file(&api_key, &image_path)
                    .with_context(|| format!("Gemini OCR failed for {}", image_path.display()))?
            }
            OcrBackend::Gemma => {
                let api_key = std::env::var("GEMMA_API_KEY")
                    .or_else(|_| std::env::var("GEMINI_API_KEY"))
                    .map_err(|_| {
                        anyhow!(
                            "Gemma OCR requires GEMMA_API_KEY or GEMINI_API_KEY. Set it before converting a PDF, e.g.: export GEMMA_API_KEY='your_api_key'"
                        )
                    })?;
                gemma::ocr_image_file(&api_key, &image_path)
                    .with_context(|| format!("Gemma OCR failed for {}", image_path.display()))?
            }
        };

        let file_name = image_path.file_name().with_context(|| {
            format!("OCR image path has no file name: {}", image_path.display())
        })?;
        let destination = image_output_dir.join(file_name);
        fs::copy(&image_path, &destination).with_context(|| {
            format!(
                "failed to copy OCR image {} to {}",
                image_path.display(),
                destination.display()
            )
        })?;
        on_page(ocr_text, destination)?;
    }

    Ok(())
}

pub fn should_use_ocr_fallback(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.is_empty() || trimmed.len() < 20
}

pub fn render_markdown(result: &PdfExtractionResult) -> String {
    let mut output = String::new();

    if !result.text.trim().is_empty() {
        output.push_str("# Extracted PDF text\n\n");
        output.push_str(result.text.trim());
        output.push_str("\n\n");
    }

    if !result.images.is_empty() {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str("## Embedded images\n\n");
        for (index, path) in result.images.iter().enumerate() {
            output.push_str(&format!("![image-{}]({})\n\n", index + 1, path.display()));
        }
    }

    output
}

fn clean_extracted_text(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

fn validate_page_range(begin_page: usize, end_page: usize) -> Result<()> {
    if begin_page == 0 || end_page == 0 || begin_page > end_page {
        return Err(anyhow!(
            "invalid page range {}-{}; pages are 1-based and begin-page must not exceed end-page",
            begin_page,
            end_page
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PdfExtractionResult, render_markdown, should_use_ocr_fallback};

    #[test]
    fn renders_markdown_with_text_content() {
        let result = PdfExtractionResult {
            text: "Hello world\nThis is a test PDF\n".to_string(),
            images: vec![],
        };

        let output = render_markdown(&result);
        assert!(output.contains("# Extracted PDF text"));
        assert!(output.contains("Hello world"));
        assert!(output.contains("This is a test PDF"));
    }

    #[test]
    fn renders_images_section_when_present() {
        let result = PdfExtractionResult {
            text: "Page one\n".to_string(),
            images: vec![std::path::PathBuf::from("assets/page-1.png")],
        };

        let output = render_markdown(&result);
        assert!(output.contains("## Embedded images"));
        assert!(output.contains("page-1.png"));
    }

    #[test]
    fn renders_images_without_text() {
        let result = PdfExtractionResult {
            text: String::new(),
            images: vec![std::path::PathBuf::from("assets/page-1.png")],
        };

        let output = render_markdown(&result);
        assert!(!output.contains("# Extracted PDF text"));
        assert!(output.contains("## Embedded images"));
        assert!(output.contains("page-1.png"));
    }

    #[test]
    fn renders_empty_result_without_sections() {
        let output = render_markdown(&PdfExtractionResult::default());

        assert!(output.is_empty());
    }

    #[test]
    fn triggers_fallback_for_blank_or_short_text() {
        assert!(should_use_ocr_fallback(" "));
        assert!(should_use_ocr_fallback("tiny"));
        assert!(!should_use_ocr_fallback(
            "This is a real extracted paragraph with enough text."
        ));
    }

    #[test]
    fn parses_ocr_backends_correctly() {
        use super::OcrBackend;
        assert_eq!("qwen".parse::<OcrBackend>().unwrap(), OcrBackend::Qwen);
        assert_eq!("qwen-coder".parse::<OcrBackend>().unwrap(), OcrBackend::Qwen);
        assert_eq!("gemini".parse::<OcrBackend>().unwrap(), OcrBackend::Gemini);
        assert_eq!("gemma".parse::<OcrBackend>().unwrap(), OcrBackend::Gemma);
        assert!("unknown".parse::<OcrBackend>().is_err());
    }

    #[test]
    fn default_backend_is_qwen() {
        use super::OcrBackend;
        assert_eq!(OcrBackend::default(), OcrBackend::Qwen);
    }
}
