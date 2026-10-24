use anyhow::{Context, Result, anyhow};
use hayro::hayro_interpret::InterpreterSettings;
use hayro::hayro_syntax::Pdf;
use std::fs;
use std::path::{Path, PathBuf};

use crate::markdown::PdfExtractionResult;

/// Validates that `begin_page` and `end_page` are 1-based and valid.
pub fn validate_page_range(begin_page: usize, end_page: usize) -> Result<()> {
    if begin_page == 0 || end_page == 0 || begin_page > end_page {
        return Err(anyhow!(
            "invalid page range {}-{}; pages are 1-based and begin-page must not exceed end-page",
            begin_page,
            end_page
        ));
    }

    Ok(())
}

/// Returns true if extracted text is too sparse or empty, requiring OCR fallback.
pub fn should_use_ocr_fallback(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.is_empty() || trimmed.len() < 20
}

/// Normalizes line endings in extracted text.
pub fn clean_extracted_text(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// Renders PDF pages into PNG images within a `pages` directory under `temp_dir`.
pub fn render_pages_to_images(
    pdf_path: &Path,
    temp_dir: &Path,
    begin_page: usize,
    end_page: usize,
) -> Result<Vec<PathBuf>> {
    let output_dir = temp_dir.join("pages");
    fs::create_dir_all(&output_dir).context("failed to create temporary page image directory")?;

    let pdf_bytes = fs::read(pdf_path)
        .with_context(|| format!("failed to read PDF for rendering: {}", pdf_path.display()))?;
    let document = Pdf::new(pdf_bytes).map_err(|error| {
        anyhow!(
            "failed to parse PDF for rendering {}: {error:?}",
            pdf_path.display()
        )
    })?;
    let cache = hayro::RenderCache::new();
    let interpreter_settings = InterpreterSettings::default();
    let render_settings = hayro::RenderSettings::default();
    let pixmap_settings = hayro::PixmapSettings {
        x_scale: 2.0,
        y_scale: 2.0,
        bg_color: hayro::vello_cpu::color::palette::css::WHITE,
    };

    let total_pages = document.pages().len();
    if end_page > total_pages {
        return Err(anyhow!(
            "page range {}-{} is outside the PDF's {} pages",
            begin_page,
            end_page,
            total_pages
        ));
    }

    let mut page_paths = Vec::with_capacity(end_page.saturating_sub(begin_page) + 1);

    for (index, page) in document
        .pages()
        .iter()
        .enumerate()
        .skip(begin_page - 1)
        .take(end_page - begin_page + 1)
    {
        let page_path = output_dir.join(format!("page-{}.png", index + 1));
        let png = hayro::render(page, &cache, &interpreter_settings, &render_settings, &pixmap_settings)
            .into_png()
            .with_context(|| format!("failed to encode rendered PDF page {}", index + 1))?;
        fs::write(&page_path, png)
            .with_context(|| format!("failed to save rendered page {}", index + 1))?;
        page_paths.push(page_path);
    }

    Ok(page_paths)
}

/// Safely extracts text by pages from PDF memory, catching and suppressing any panics
/// caused by malformed/unsupported font encoding tables inside pdf-extract.
pub fn safe_extract_text_by_pages(bytes: &[u8]) -> Option<Vec<String>> {
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pdf_extract::extract_text_from_mem_by_pages(bytes).ok()
    }));
    std::panic::set_hook(prev_hook);
    res.ok().flatten()
}

/// Determines the total number of pages in a PDF file quickly without rasterizing pages.
pub fn get_pdf_page_count(pdf_path: &Path) -> Result<usize> {
    let pdf_bytes = fs::read(pdf_path)
        .with_context(|| format!("failed to read PDF file {}", pdf_path.display()))?;

    if let Ok(document) = Pdf::new(pdf_bytes.clone()) {
        let count = document.pages().len();
        if count > 0 {
            return Ok(count);
        }
    }

    if let Some(pages) = safe_extract_text_by_pages(&pdf_bytes)
        && !pages.is_empty()
    {
        return Ok(pages.len());
    }

    Err(anyhow!(
        "failed to determine page count for PDF {}",
        pdf_path.display()
    ))
}

/// Attempts to extract plain text directly from the PDF streams without OCR.
pub fn extract_pdf_text(
    input_path: &Path,
    begin_page: usize,
    end_page: usize,
) -> Result<PdfExtractionResult> {
    validate_page_range(begin_page, end_page)?;

    let bytes = fs::read(input_path)
        .with_context(|| format!("failed to read PDF file {}", input_path.display()))?;

    let pages = safe_extract_text_by_pages(&bytes).ok_or_else(|| {
        anyhow!(
            "failed to parse PDF text streams in {}",
            input_path.display()
        )
    })?;
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
        return Err(anyhow!(
            "No readable text was found in the PDF; the scanned-PDF OCR fallback is required."
        ));
    }

    Ok(PdfExtractionResult {
        text: cleaned,
        images: Vec::new(),
        figures: Vec::new(),
    })
}
