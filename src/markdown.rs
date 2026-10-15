use crate::figure::CroppedFigure;
use std::path::PathBuf;

/// Results of a document conversion or text extraction run.
#[derive(Debug, Default, Clone)]
pub struct PdfExtractionResult {
    pub text: String,
    pub images: Vec<PathBuf>,
    pub figures: Vec<CroppedFigure>,
}

/// Optional configuration for conversion runs.
#[derive(Debug, Default, Clone)]
pub struct ConversionConfig {
    pub temp_dir: Option<PathBuf>,
    pub output_images_dir: Option<PathBuf>,
}

/// Formats the final verbatim Markdown document from extraction results.
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

/// Injects detected figure image links into the appropriate locations of the page text.
pub fn attach_figures_to_page_text(page_text: &str, figures: &[CroppedFigure]) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

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
            images: vec![PathBuf::from("assets/page-1.png")],
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
            images: vec![PathBuf::from("assets/page-1.png")],
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
    fn renders_figures_with_exact_labels() {
        let result = PdfExtractionResult {
            text: "# Section\nSome text".to_string(),
            images: vec![],
            figures: vec![CroppedFigure {
                figure_label: "Figure VII-3".to_string(),
                caption: None,
                filename: "Figure-VII-3.png".to_string(),
                file_path: PathBuf::from("output/images/Figure-VII-3.png"),
            }],
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
            figures: vec![CroppedFigure {
                figure_label: "Figure 4.7".to_string(),
                caption: Some("Variation of species mole fractions with distance.".to_string()),
                filename: "Figure-4-7.png".to_string(),
                file_path: PathBuf::from("output/images/Figure-4-7.png"),
            }],
        };

        let output = render_markdown(&result);
        assert!(output.contains("## Figures"));
        assert!(output.contains("![Figure 4.7](images/Figure-4-7.png)"));
        assert!(
            output.contains("*Figure 4.7. Variation of species mole fractions with distance.*")
        );
    }

    #[test]
    fn test_attach_figures_to_page_text() {
        let figures = vec![CroppedFigure {
            figure_label: "Figure 4.7".to_string(),
            caption: Some("Variation of species mole fractions with distance.".to_string()),
            filename: "Figure-4-7.png".to_string(),
            file_path: PathBuf::from("output/images/Figure-4-7.png"),
        }];

        // Case 1: Figure label is present in page text
        let page_text =
            "Introductory paragraph.\n\nFigure 4.7.—Variation of species.\n\nNext paragraph.";
        let attached = attach_figures_to_page_text(page_text, &figures);
        assert!(
            attached.contains(
                "![Figure 4.7](images/Figure-4-7.png)\n\nFigure 4.7.—Variation of species."
            )
        );

        // Case 2: Figure label is absent from page text
        let page_text_no_label = "Page with only text paragraphs.";
        let attached2 = attach_figures_to_page_text(page_text_no_label, &figures);
        assert!(attached2.contains("![Figure 4.7](images/Figure-4-7.png)"));
        assert!(
            attached2.contains("*Figure 4.7. Variation of species mole fractions with distance.*")
        );
    }
}
