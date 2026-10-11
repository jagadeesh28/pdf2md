pub mod fallback;
pub mod figure;
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
    pub figures: Vec<figure::CroppedFigure>,
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
        |text, figures| {
            if !text.trim().is_empty() {
                extracted.text.push_str(&text);
                if !extracted.text.ends_with("\n\n") {
                    extracted.text.push_str("\n\n");
                }
            }
            for fig in figures {
                extracted.images.push(fig.file_path.clone());
                extracted.figures.push(fig);
            }

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
        figures: Vec::new(),
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
        |text, figures| {
            if !text.trim().is_empty() {
                result.text.push_str(&text);
                result.text.push_str("\n\n");
            }
            for fig in figures {
                result.images.push(fig.file_path.clone());
                result.figures.push(fig);
            }
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
    F: FnMut(String, Vec<figure::CroppedFigure>) -> Result<()>,
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
    F: FnMut(String, Vec<figure::CroppedFigure>) -> Result<()>,
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

    let pdf_bytes = fs::read(input_path).ok();
    let pdf_pages = pdf_bytes.as_ref().and_then(|bytes| {
        pdf_extract::extract_text_from_mem_by_pages(bytes).ok()
    });

    let concurrency = resolve_concurrency();
    let total_pages = page_images.len();

    // Prepare page items with metadata
    let mut page_items = Vec::with_capacity(total_pages);
    for (page_offset, image_path) in page_images.into_iter().enumerate() {
        let page_num = begin_page + page_offset;
        let raw_page_text = pdf_pages
            .as_ref()
            .and_then(|pages| pages.get(page_num - 1))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        page_items.push((page_num, image_path, raw_page_text));
    }

    if concurrency <= 1 || total_pages <= 1 {
        // Sequential execution
        for (page_num, image_path, raw_page_text) in page_items {
            let (ocr_text, cropped_figures) = process_single_page(
                backend,
                &image_path,
                image_output_dir,
                page_num,
                raw_page_text.as_deref(),
            )?;
            on_page(ocr_text, cropped_figures)?;
        }
    } else {
        // Concurrent worker execution preserving exact page order
        use std::collections::BTreeMap;
        use std::sync::mpsc;
        use std::thread;

        println!(
            "Processing {total_pages} pages concurrently with concurrency={concurrency} (maintaining exact page order)..."
        );

        let (work_tx, work_rx) = std::sync::mpsc::channel();
        for item in page_items {
            let _ = work_tx.send(item);
        }
        drop(work_tx); // Close queue so workers terminate when all pages are consumed

        let (result_tx, result_rx) = mpsc::channel();

        let mut worker_handles = Vec::with_capacity(concurrency);
        let work_rx = std::sync::Arc::new(std::sync::Mutex::new(work_rx));

        for _ in 0..concurrency {
            let work_rx_clone = std::sync::Arc::clone(&work_rx);
            let result_tx_clone = result_tx.clone();
            let img_out_dir = image_output_dir.to_path_buf();

            let handle = thread::spawn(move || {
                loop {
                    let item = {
                        let Ok(rx) = work_rx_clone.lock() else { break };
                        rx.recv().ok()
                    };

                    let Some((page_num, image_path, raw_page_text)) = item else {
                        break;
                    };

                    let page_res = process_single_page(
                        backend,
                        &image_path,
                        &img_out_dir,
                        page_num,
                        raw_page_text.as_deref(),
                    );

                    if result_tx_clone.send((page_num, page_res)).is_err() {
                        break;
                    }
                }
            });
            worker_handles.push(handle);
        }
        drop(result_tx); // Drop extra sender so rx closes when workers complete

        // Stream results to on_page strictly in ascending page order (begin_page..=end_page)
        let mut pending_results: BTreeMap<usize, (String, Vec<figure::CroppedFigure>)> = BTreeMap::new();
        let mut next_expected_page = begin_page;

        while let Ok((page_num, page_res)) = result_rx.recv() {
            let (ocr_text, figures) = page_res?;
            pending_results.insert(page_num, (ocr_text, figures));

            while let Some((&page, _)) = pending_results.iter().next() {
                if page == next_expected_page {
                    let (_, (ocr_text, figures)) = pending_results.remove_entry(&page).unwrap();
                    on_page(ocr_text, figures)?;
                    next_expected_page += 1;
                } else {
                    break;
                }
            }
        }

        // Wait for all workers to cleanly exit
        for handle in worker_handles {
            let _ = handle.join();
        }

        // Emit any remaining pages in order
        while let Some((page, (ocr_text, figures))) = pending_results.pop_first() {
            if page == next_expected_page {
                next_expected_page += 1;
            }
            on_page(ocr_text, figures)?;
        }
    }

    Ok(())
}

fn process_single_page(
    backend: OcrBackend,
    image_path: &Path,
    image_output_dir: &Path,
    page_num: usize,
    raw_page_text: Option<&str>,
) -> Result<(String, Vec<figure::CroppedFigure>)> {
    let ocr_text = match backend {
        OcrBackend::Qwen => match qwen::ocr_image_file(image_path) {
            Ok(text) => text,
            Err(err) => {
                let err_msg = err.to_string();
                if err_msg.contains("does not support multimodal") {
                    if let Some(txt) = raw_page_text {
                        eprintln!(
                            "Notice: Local Qwen model is text-only; formatting extracted page {page_num} text into Obsidian Markdown..."
                        );
                        qwen::transcribe_text(txt).with_context(|| {
                            format!("failed to format extracted page {page_num} text with Qwen")
                        })?
                    } else {
                        return Err(err);
                    }
                } else if err_msg.contains("token repeat limit reached")
                    || err_msg.contains("repetitive token loop")
                {
                    if let Some(txt) = raw_page_text {
                        eprintln!(
                            "Notice: Page {page_num} vision OCR entered a repetitive token loop; formatting extracted PDF text into Markdown..."
                        );
                        match qwen::transcribe_text(txt) {
                            Ok(formatted) => formatted,
                            Err(_) => txt.to_string(),
                        }
                    } else {
                        eprintln!(
                            "Warning: Page {page_num} vision OCR entered a repetitive token loop and no raw text was available in the PDF. Emitting placeholder note so document conversion continues."
                        );
                        format!(
                            "> [!WARNING] Page {page_num} Transcription Incomplete\n\
                             > Local vision model entered a repetitive token loop on this page and generation was stopped by Ollama.\n"
                        )
                    }
                } else {
                    return Err(err);
                }
            }
        },
        OcrBackend::Gemini => {
            let api_key = std::env::var("GEMINI_API_KEY").map_err(|_| {
                anyhow!(
                    "Gemini OCR requires GEMINI_API_KEY. Set it before converting a PDF, e.g.: export GEMINI_API_KEY='your_api_key'"
                )
            })?;
            gemini::ocr_image_file(&api_key, image_path)
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
            gemma::ocr_image_file(&api_key, image_path)
                .with_context(|| format!("Gemma OCR failed for {}", image_path.display()))?
        }
    };

    // Detect and crop figures on this page using Qwen Vision into image_output_dir
    let cropped_figures = figure::detect_and_crop_figures(image_path, image_output_dir, page_num)
        .unwrap_or_else(|err| {
            eprintln!("Figure detection skipped for page {page_num}: {err:#}");
            Vec::new()
        });

    Ok((ocr_text, cropped_figures))
}

pub fn resolve_concurrency() -> usize {
    if let Ok(val) = std::env::var("PDF2MD_CONCURRENCY").or_else(|_| std::env::var("CONCURRENCY"))
        && let Ok(parsed) = val.parse::<usize>()
    {
        return parsed.max(1);
    }
    // Default to 2 concurrent page workers to achieve 2x speedup without overloading VRAM
    2
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

    if !result.figures.is_empty() {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str("## Figures\n\n");
        for fig in &result.figures {
            output.push_str(&format!(
                "![{}](images/{})\n*{}*\n\n",
                fig.figure_label, fig.filename, fig.figure_label
            ));
        }
    } else if !result.images.is_empty() {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str("## Embedded images\n\n");
        for (index, path) in result.images.iter().enumerate() {
            let path_str = path.to_string_lossy().replace('\\', "/");
            output.push_str(&format!("![image-{}]({})\n\n", index + 1, path_str));
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
            figures: vec![],
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
            figures: vec![],
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
            figures: vec![],
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

    #[test]
    fn renders_figures_with_exact_labels() {
        let result = PdfExtractionResult {
            text: "# Section\nSome text".to_string(),
            images: vec![],
            figures: vec![
                super::figure::CroppedFigure {
                    figure_label: "Figure VII-3".to_string(),
                    filename: "Figure-VII-3.png".to_string(),
                    file_path: std::path::PathBuf::from("output/images/Figure-VII-3.png"),
                },
            ],
        };

        let output = render_markdown(&result);
        assert!(output.contains("## Figures"));
        assert!(output.contains("![Figure VII-3](images/Figure-VII-3.png)"));
        assert!(output.contains("*Figure VII-3*"));
    }

    #[test]
    fn test_resolve_concurrency() {
        assert!(super::resolve_concurrency() >= 1);
    }
}
