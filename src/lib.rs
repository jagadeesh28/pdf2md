pub mod backends;
pub mod batch;
pub mod cli;
pub mod fallback;
pub mod figure;
pub mod figure_analysis;
pub mod markdown;
pub mod pdf;
pub mod pipeline;
pub mod preprocess;

// Backwards-compatible module re-exports
pub use backends::gemini;
pub use backends::gemma;
pub use backends::qwen;

// Public API re-exports
pub use backends::{OcrBackend, resolve_default_backend};
pub use batch::{
    BatchConversionResult, DEFAULT_MAX_SEARCH_DEPTH, collect_pdf_files,
    collect_pdf_files_with_depth, process_pdf_directory, resolve_max_search_depth,
    resolve_unique_output_dir, sanitize_folder_name,
};
pub use fallback::get_pdf_page_count;
pub use markdown::{
    ConversionConfig, PdfExtractionResult, attach_figures_to_page_text, render_markdown,
};
pub use pdf::{
    clean_extracted_text, extract_pdf_text, should_use_ocr_fallback, validate_page_range,
};
pub use pipeline::{
    fall_back_to_ocr, fall_back_to_ocr_with_backend, process_ocr_pages,
    process_ocr_pages_with_backend, process_pdf, process_pdf_with_backend, resolve_concurrency,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ocr_backends_correctly() {
        assert_eq!("qwen".parse::<OcrBackend>().unwrap(), OcrBackend::Qwen);
        assert_eq!(
            "qwen-coder".parse::<OcrBackend>().unwrap(),
            OcrBackend::Qwen
        );
        assert_eq!("gemini".parse::<OcrBackend>().unwrap(), OcrBackend::Gemini);
        assert_eq!("gemma".parse::<OcrBackend>().unwrap(), OcrBackend::Gemma);
        assert!("unknown".parse::<OcrBackend>().is_err());
    }

    #[test]
    fn default_backend_is_qwen() {
        assert_eq!(OcrBackend::default(), OcrBackend::Qwen);
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
    fn test_get_pdf_page_count() {
        let p = std::path::Path::new("input/CEA/CC_code.pdf");
        if p.exists() {
            let count = get_pdf_page_count(p).unwrap();
            assert!(count > 0);
        }
    }
}
