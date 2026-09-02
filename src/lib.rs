mod fallback;
mod gemini;

use anyhow::{Context, Result, anyhow};
use std::fs;
use std::path::{Path, PathBuf};

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
    validate_page_range(begin_page, end_page)?;

    let extracted = match extract_pdf_text(input_path, begin_page, end_page) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("Direct extraction failed: {err}. Trying OCR fallback.");
            let image_output_dir = output_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("images");
            fall_back_to_ocr(input_path, &image_output_dir, begin_page, end_page)
                .with_context(|| format!("OCR fallback failed for {}", input_path.display()))?
        }
    };

    let markdown = render_markdown(&extracted);
    let output_dir = output_path
        .parent()
        .unwrap_or_else(|| Path::new("."));

    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create output directory {}", output_dir.display()))?;
    fs::write(output_path, markdown)
        .with_context(|| format!("failed to write markdown output to {}", output_path.display()))?;

    println!("Converted {} to {}", input_path.display(), output_path.display());
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
    validate_page_range(begin_page, end_page)?;

    let temp_dir = tempfile::tempdir().context("failed to create temporary OCR directory")?;
    let page_images = fallback::render_pages_to_images(
        input_path,
        temp_dir.path(),
        begin_page,
        end_page,
    )
        .with_context(|| format!("failed to rasterize PDF {} for OCR", input_path.display()))?;

    fs::create_dir_all(image_output_dir).with_context(|| {
        format!(
            "failed to create image output directory {}",
            image_output_dir.display()
        )
    })?;
    
    let api_key = std::env::var("GEMINI_API_KEY").map_err(|_| {
        anyhow::anyhow!(
            "OCR fallback requires GEMINI_API_KEY. Set it before running the scanned-PDF path, e.g. on Windows: $env:GEMINI_API_KEY='your_api_key'"
        )
    })?;
    
    let mut text_parts = Vec::new();
    let mut images = Vec::new();
    for image_path in page_images {
        let ocr_text = gemini::ocr_image_file(&api_key, &image_path)
            .with_context(|| format!("Gemini OCR failed for {}", image_path.display()))?;
        if !ocr_text.trim().is_empty() {
            text_parts.push(ocr_text);
        }
        let file_name = image_path
            .file_name()
            .with_context(|| format!("OCR image path has no file name: {}", image_path.display()))?;
        let destination = image_output_dir.join(file_name);
        fs::copy(&image_path, &destination).with_context(|| {
            format!(
                "failed to copy OCR image {} to {}",
                image_path.display(),
                destination.display()
            )
        })?;
        images.push(destination);
    }
    
    Ok(PdfExtractionResult {
        text: text_parts.join("\n\n"),
        images,
    })
}

pub fn should_use_ocr_fallback(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.is_empty() || trimmed.len() < 20
}

pub fn render_markdown(result: &PdfExtractionResult) -> String {
    let mut output = String::new();

    if !result.text.trim().is_empty() {
        output.push_str("# Extracted PDF text\n\n");
        for line in result.text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                output.push('\n');
                continue;
            }

            output.push_str(trimmed);
            output.push('\n');
        }
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
        assert!(!should_use_ocr_fallback("This is a real extracted paragraph with enough text."));
    }
}
