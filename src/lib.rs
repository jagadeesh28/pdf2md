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

pub use fallback::get_pdf_page_count;

/// Sanitizes a string into a safe folder name across platforms.
pub fn sanitize_folder_name(name: &str) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return "document".to_string();
    }
    let sanitized: String = trimmed
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => c,
        })
        .collect();
    let res = sanitized.trim();
    if res.is_empty() {
        "document".to_string()
    } else {
        res.to_string()
    }
}

/// Collects all PDF files (.pdf, case-insensitive) in a directory, sorted by name.
pub fn collect_pdf_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut pdfs = Vec::new();
    let entries = fs::read_dir(dir)
        .with_context(|| format!("failed to read directory {}", dir.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext.eq_ignore_ascii_case("pdf") {
                    pdfs.push(path);
                }
            }
        }
    }
    pdfs.sort();
    Ok(pdfs)
}

/// Finds a unique folder path inside base_dir for the given folder_name.
pub fn resolve_unique_output_dir(base_dir: &Path, folder_name: &str) -> PathBuf {
    let candidate = base_dir.join(folder_name);
    if !candidate.exists() {
        return candidate;
    }
    // If it exists and is an empty directory, reuse it
    if candidate.is_dir() {
        if let Ok(mut entries) = fs::read_dir(&candidate) {
            if entries.next().is_none() {
                return candidate;
            }
        }
    }
    let mut counter = 1;
    loop {
        let next = base_dir.join(format!("{}_{counter}", folder_name));
        if !next.exists() {
            return next;
        }
        if next.is_dir() {
            if let Ok(mut entries) = fs::read_dir(&next) {
                if entries.next().is_none() {
                    return next;
                }
            }
        }
        counter += 1;
    }
}

#[derive(Debug, Default)]
pub struct BatchConversionResult {
    pub total: usize,
    pub succeeded: usize,
    pub failed: Vec<(PathBuf, String)>,
}

/// Converts all PDF files in input_dir one by one, outputting each into a uniquely named folder in output_base_dir.
pub fn process_pdf_directory(
    input_dir: &Path,
    output_base_dir: &Path,
    page_range: Option<(usize, usize)>,
    backend: OcrBackend,
) -> Result<BatchConversionResult> {
    let pdf_files = collect_pdf_files(input_dir)?;
    if pdf_files.is_empty() {
        return Err(anyhow!(
            "No PDF files (.pdf) found in directory '{}'",
            input_dir.display()
        ));
    }

    fs::create_dir_all(output_base_dir).with_context(|| {
        format!(
            "failed to create output base directory {}",
            output_base_dir.display()
        )
    })?;

    println!(
        "\nFound {} PDF files in '{}'. Processing one by one...",
        pdf_files.len(),
        input_dir.display()
    );

    let mut result = BatchConversionResult {
        total: pdf_files.len(),
        succeeded: 0,
        failed: Vec::new(),
    };

    let mut used_folder_names = std::collections::HashSet::new();

    for (index, pdf_path) in pdf_files.iter().enumerate() {
        let pdf_stem = pdf_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("document");
        let base_folder_name = sanitize_folder_name(pdf_stem);

        // Ensure unique folder name within this batch run
        let mut folder_name = base_folder_name.clone();
        let mut counter = 1;
        while used_folder_names.contains(&folder_name) {
            folder_name = format!("{base_folder_name}_{counter}");
            counter += 1;
        }
        used_folder_names.insert(folder_name.clone());

        let pdf_output_dir = output_base_dir.join(&folder_name);
        if let Err(err) = fs::create_dir_all(&pdf_output_dir) {
            eprintln!(
                "Failed to create output directory {}: {err:#}",
                pdf_output_dir.display()
            );
            result.failed.push((pdf_path.clone(), err.to_string()));
            continue;
        }

        let target_md_path = pdf_output_dir.join(format!("{folder_name}.md"));

        let total_pages = match fallback::get_pdf_page_count(pdf_path) {
            Ok(count) => count,
            Err(err) => {
                eprintln!(
                    "Failed to read page count for {}: {err:#}",
                    pdf_path.display()
                );
                result.failed.push((pdf_path.clone(), err.to_string()));
                continue;
            }
        };

        let (begin_page, end_page) = match page_range {
            Some((start, end)) => {
                if start > total_pages {
                    eprintln!(
                        "Warning: Skipping {} (start page {start} exceeds total pages {total_pages})",
                        pdf_path.display()
                    );
                    result.failed.push((
                        pdf_path.clone(),
                        format!("start page {start} exceeds total pages {total_pages}"),
                    ));
                    continue;
                }
                (start, end.min(total_pages))
            }
            None => (1, total_pages),
        };

        println!(
            "\n=======================================================\n\
             [{}/{}] Converting '{}'\n\
             Pages: {}..{} (total: {})\n\
             Output directory: {}\n\
             Markdown: {}\n\
             Images: {}\n\
             =======================================================",
            index + 1,
            pdf_files.len(),
            pdf_path.file_name().and_then(|s| s.to_str()).unwrap_or("document.pdf"),
            begin_page,
            end_page,
            total_pages,
            pdf_output_dir.display(),
            target_md_path.display(),
            pdf_output_dir.join("images").display()
        );

        match process_pdf_with_backend(
            pdf_path,
            &target_md_path,
            begin_page,
            end_page,
            backend,
        ) {
            Ok(()) => {
                result.succeeded += 1;
                println!(
                    "✓ Successfully converted '{}' -> {}",
                    pdf_path.file_name().and_then(|s| s.to_str()).unwrap_or("document.pdf"),
                    target_md_path.display()
                );
            }
            Err(err) => {
                eprintln!(
                    "✗ Failed to convert '{}': {err:#}",
                    pdf_path.display()
                );
                result.failed.push((pdf_path.clone(), err.to_string()));
            }
        }
    }

    println!("\n================ Batch Conversion Summary ================");
    println!(
        "Total PDFs: {}, Succeeded: {}, Failed: {}",
        result.total,
        result.succeeded,
        result.failed.len()
    );
    if !result.failed.is_empty() {
        println!("\nFailed documents:");
        for (path, err) in &result.failed {
            println!("  - {}: {}", path.display(), err);
        }
    }
    println!("==========================================================");

    Ok(result)
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

    let ocr_text = attach_figures_to_page_text(&ocr_text, &cropped_figures);

    Ok((ocr_text, cropped_figures))
}

pub fn attach_figures_to_page_text(
    page_text: &str,
    figures: &[figure::CroppedFigure],
) -> String {
    if figures.is_empty() {
        return page_text.to_string();
    }

    let mut result = page_text.to_string();

    for fig in figures {
        let image_link = format!("![{}](images/{})", fig.figure_label, fig.filename);
        if result.contains(&image_link) || result.contains(&fig.filename) {
            continue;
        }

        let caption_text = match &fig.caption {
            Some(cap) if !cap.trim().is_empty() => {
                let trimmed = cap.trim();
                let lower_label = fig.figure_label.to_lowercase();
                let lower_cap = trimmed.to_lowercase();
                if lower_cap.starts_with(&lower_label) {
                    trimmed.to_string()
                } else {
                    format!("{}. {}", fig.figure_label, trimmed)
                }
            }
            _ => fig.figure_label.clone(),
        };

        let lower_text = result.to_lowercase();
        let lower_label = fig.figure_label.to_lowercase();

        if let Some(pos) = lower_text.find(&lower_label) {
            let line_start = result[..pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
            result.insert_str(line_start, &format!("{image_link}\n\n"));
        } else {
            if !result.is_empty() && !result.ends_with("\n\n") {
                if result.ends_with('\n') {
                    result.push('\n');
                } else {
                    result.push_str("\n\n");
                }
            }
            result.push_str(&format!("{image_link}\n\n*{caption_text}*\n\n"));
        }
    }

    result
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
            let caption_text = match &fig.caption {
                Some(cap) if !cap.trim().is_empty() => {
                    let trimmed = cap.trim();
                    let lower_label = fig.figure_label.to_lowercase();
                    let lower_cap = trimmed.to_lowercase();
                    if lower_cap.starts_with(&lower_label) {
                        trimmed.to_string()
                    } else {
                        format!("{}. {}", fig.figure_label, trimmed)
                    }
                }
                _ => fig.figure_label.clone(),
            };
            output.push_str(&format!(
                "![{}](images/{})\n*{}*\n\n",
                fig.figure_label, fig.filename, caption_text
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
                    caption: None,
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
    fn renders_figures_with_caption() {
        let result = PdfExtractionResult {
            text: "# Section\nSome text".to_string(),
            images: vec![],
            figures: vec![
                super::figure::CroppedFigure {
                    figure_label: "Figure 4.7".to_string(),
                    caption: Some("Variation of species mole fractions with distance.".to_string()),
                    filename: "Figure-4-7.png".to_string(),
                    file_path: std::path::PathBuf::from("output/images/Figure-4-7.png"),
                },
            ],
        };

        let output = render_markdown(&result);
        assert!(output.contains("## Figures"));
        assert!(output.contains("![Figure 4.7](images/Figure-4-7.png)"));
        assert!(output.contains("*Figure 4.7. Variation of species mole fractions with distance.*"));
    }

    #[test]
    fn test_attach_figures_to_page_text() {
        let figures = vec![
            super::figure::CroppedFigure {
                figure_label: "Figure 4.7".to_string(),
                caption: Some("Variation of species mole fractions with distance.".to_string()),
                filename: "Figure-4-7.png".to_string(),
                file_path: std::path::PathBuf::from("output/images/Figure-4-7.png"),
            },
        ];

        // Case 1: Figure label is present in page text
        let page_text = "Introductory paragraph.\n\nFigure 4.7.—Variation of species.\n\nNext paragraph.";
        let attached = super::attach_figures_to_page_text(page_text, &figures);
        assert!(attached.contains("![Figure 4.7](images/Figure-4-7.png)\n\nFigure 4.7.—Variation of species."));

        // Case 2: Figure label is absent from page text
        let page_text_no_label = "Page with only text paragraphs.";
        let attached2 = super::attach_figures_to_page_text(page_text_no_label, &figures);
        assert!(attached2.contains("![Figure 4.7](images/Figure-4-7.png)"));
        assert!(attached2.contains("*Figure 4.7. Variation of species mole fractions with distance.*"));
    }

    #[test]
    fn test_resolve_concurrency() {
        assert!(super::resolve_concurrency() >= 1);
    }

    #[test]
    fn test_sanitize_folder_name() {
        assert_eq!(super::sanitize_folder_name("CC_code"), "CC_code");
        assert_eq!(super::sanitize_folder_name("TB_26-04 FInal"), "TB_26-04 FInal");
        assert_eq!(super::sanitize_folder_name("doc:name*test?"), "doc_name_test_");
        assert_eq!(super::sanitize_folder_name("   "), "document");
    }

    #[test]
    fn test_resolve_unique_output_dir() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path();

        let dir1 = super::resolve_unique_output_dir(base, "doc");
        assert_eq!(dir1, base.join("doc"));
        std::fs::create_dir_all(&dir1).unwrap();
        // Create a file inside dir1 so it's not empty
        std::fs::write(dir1.join("file.txt"), "hello").unwrap();

        let dir2 = super::resolve_unique_output_dir(base, "doc");
        assert_eq!(dir2, base.join("doc_1"));
    }

    #[test]
    fn test_collect_pdf_files() {
        let p = std::path::Path::new("input/CEA");
        if p.exists() {
            let files = super::collect_pdf_files(p).unwrap();
            assert!(!files.is_empty());
            assert!(files.iter().all(|f| f.extension().map_or(false, |ext| ext.eq_ignore_ascii_case("pdf"))));
        }
    }

    #[test]
    fn test_get_pdf_page_count() {
        let p = std::path::Path::new("input/CEA/CC_code.pdf");
        if p.exists() {
            let count = super::get_pdf_page_count(p).unwrap();
            assert!(count > 0);
        }
    }
}
