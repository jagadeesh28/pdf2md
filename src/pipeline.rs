use anyhow::{Context, Result, anyhow};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use crate::backends::{OcrBackend, resolve_default_backend};
use crate::backends::{gemini, gemma, qwen};
use crate::figure::{self, CroppedFigure};
use crate::markdown::{PdfExtractionResult, attach_figures_to_page_text, render_markdown};
use crate::pdf::{render_pages_to_images, validate_page_range};

/// Resolves worker concurrency count from `PDF2MD_CONCURRENCY` or `CONCURRENCY` env vars (default: 2).
pub fn resolve_concurrency() -> usize {
    if let Ok(val) = std::env::var("PDF2MD_CONCURRENCY").or_else(|_| std::env::var("CONCURRENCY"))
        && let Ok(parsed) = val.parse::<usize>()
    {
        return parsed.max(1);
    }
    // Default to 2 concurrent page workers now that Ollama is configured with OLLAMA_NUM_PARALLEL=2 on the L4 GPU
    2
}

/// Converts a single PDF file using the default backend.
pub fn process_pdf(
    input_path: &Path,
    output_path: &Path,
    begin_page: usize,
    end_page: usize,
) -> Result<()> {
    let backend = resolve_default_backend();
    process_pdf_with_backend(input_path, output_path, begin_page, end_page, backend)
}

/// Converts a single PDF file using the specified OCR backend.
pub fn process_pdf_with_backend(
    input_path: &Path,
    output_path: &Path,
    begin_page: usize,
    end_page: usize,
    backend: OcrBackend,
) -> Result<()> {
    validate_page_range(begin_page, end_page)?;

    let pdf_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("document");
    let images_folder_name = format!("{pdf_stem}_images");
    let image_output_dir = output_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&images_folder_name);
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

/// Fallback OCR conversion using default backend.
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

/// Fallback OCR conversion using specified backend.
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

/// Streams processed pages using the default backend.
pub fn process_ocr_pages<F>(
    input_path: &Path,
    image_output_dir: &Path,
    begin_page: usize,
    end_page: usize,
    on_page: F,
) -> Result<()>
where
    F: FnMut(String, Vec<CroppedFigure>) -> Result<()>,
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

/// Concurrent page processing worker engine that guarantees exact ascending page ordering.
pub fn process_ocr_pages_with_backend<F>(
    input_path: &Path,
    image_output_dir: &Path,
    begin_page: usize,
    end_page: usize,
    backend: OcrBackend,
    mut on_page: F,
) -> Result<()>
where
    F: FnMut(String, Vec<CroppedFigure>) -> Result<()>,
{
    validate_page_range(begin_page, end_page)?;

    let temp_dir = tempfile::tempdir().context("failed to create temporary OCR directory")?;
    let page_images = render_pages_to_images(input_path, temp_dir.path(), begin_page, end_page)
        .with_context(|| format!("failed to rasterize PDF {} for OCR", input_path.display()))?;

    fs::create_dir_all(image_output_dir).with_context(|| {
        format!(
            "failed to create image output directory {}",
            image_output_dir.display()
        )
    })?;

    let pdf_bytes = fs::read(input_path).ok();
    let pdf_pages = pdf_bytes
        .as_ref()
        .and_then(|bytes| crate::pdf::safe_extract_text_by_pages(bytes));

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
        println!(
            "Processing {total_pages} pages concurrently with concurrency={concurrency} (maintaining exact page order)..."
        );

        let (work_tx, work_rx) = mpsc::channel();
        for item in page_items {
            let _ = work_tx.send(item);
        }
        drop(work_tx); // Close queue so workers terminate when all pages are consumed

        let (result_tx, result_rx) = mpsc::channel();

        let mut worker_handles = Vec::with_capacity(concurrency);
        let work_rx = Arc::new(Mutex::new(work_rx));

        for _ in 0..concurrency {
            let work_rx_clone = Arc::clone(&work_rx);
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
        let mut pending_results: BTreeMap<usize, (String, Vec<figure::CroppedFigure>)> =
            BTreeMap::new();
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
                    if let Some(txt) = raw_page_text {
                        eprintln!(
                            "Warning: Page {page_num} vision OCR failed ({err_msg}); falling back to extracted PDF text..."
                        );
                        match qwen::transcribe_text(txt) {
                            Ok(formatted) => formatted,
                            Err(_) => txt.to_string(),
                        }
                    } else {
                        eprintln!(
                            "Warning: Page {page_num} vision OCR failed ({err_msg}) and no raw text was available in the PDF. Emitting placeholder note so document conversion continues."
                        );
                        format!(
                            "> [!WARNING] Page {page_num} Transcription Incomplete\n\
                             > Local vision model failed to transcribe this page: {err_msg}\n"
                        )
                    }
                }
            }
        },
        OcrBackend::Gemini => {
            let api_key = std::env::var("GEMINI_API_KEY").map_err(|_| {
                anyhow!(
                    "Gemini OCR requires GEMINI_API_KEY. Set it before converting a PDF, e.g.: export GEMINI_API_KEY='your_api_key'"
                )
            })?;
            match gemini::ocr_image_file(&api_key, image_path) {
                Ok(text) => text,
                Err(err) => {
                    if let Some(txt) = raw_page_text {
                        eprintln!(
                            "Notice: Gemini OCR failed for page {page_num} ({err:#}); falling back to extracted PDF text..."
                        );
                        txt.to_string()
                    } else {
                        eprintln!(
                            "Warning: Gemini OCR failed for page {page_num} ({err:#}) and no raw text was available in the PDF."
                        );
                        format!(
                            "> [!WARNING] Page {page_num} Transcription Incomplete\n\
                             > Gemini vision OCR failed on this page: {err:#}\n"
                        )
                    }
                }
            }
        }
        OcrBackend::Gemma => {
            let api_key = std::env::var("GEMMA_API_KEY")
                .or_else(|_| std::env::var("GEMINI_API_KEY"))
                .map_err(|_| {
                    anyhow!(
                        "Gemma OCR requires GEMMA_API_KEY or GEMINI_API_KEY. Set it before converting a PDF, e.g.: export GEMMA_API_KEY='your_api_key'"
                    )
                })?;
            match gemma::ocr_image_file(&api_key, image_path) {
                Ok(text) => text,
                Err(err) => {
                    if let Some(txt) = raw_page_text {
                        eprintln!(
                            "Notice: Gemma OCR failed for page {page_num} ({err:#}); falling back to extracted PDF text..."
                        );
                        txt.to_string()
                    } else {
                        eprintln!(
                            "Warning: Gemma OCR failed for page {page_num} ({err:#}) and no raw text was available in the PDF."
                        );
                        format!(
                            "> [!WARNING] Page {page_num} Transcription Incomplete\n\
                             > Gemma vision OCR failed on this page: {err:#}\n"
                        )
                    }
                }
            }
        }
    };

    // Detect and crop figures on this page using the active backend vision model into image_output_dir
    let cropped_figures =
        figure::detect_and_crop_figures(backend, image_path, image_output_dir, page_num)
            .unwrap_or_else(|err| {
                eprintln!("Figure detection skipped for page {page_num}: {err:#}");
                Vec::new()
            });

    let ocr_text = attach_figures_to_page_text(&ocr_text, &cropped_figures);

    Ok((ocr_text, cropped_figures))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_concurrency() {
        assert!(resolve_concurrency() >= 1);
    }
}
