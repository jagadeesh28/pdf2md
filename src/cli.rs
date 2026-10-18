use std::env;
use std::fs;
use std::path::Path;
use std::process;

use crate::backends::{OcrBackend, resolve_default_backend};
use crate::batch::{process_pdf_directory, sanitize_folder_name};
use crate::pdf::get_pdf_page_count;
use crate::pipeline::process_pdf_with_backend;

pub fn print_usage() {
    eprintln!(
        r#"Usage:
    # Single PDF conversion:
    pdf2md <input.pdf> <output.md> [begin-page] [end-page] [backend]
    pdf2md <input.pdf> <output-dir> [begin-page] [end-page] [backend]

    # Batch directory conversion:
    pdf2md <input-dir> <output-dir> [begin-page] [end-page] [backend]

    # Normalize LaTeX math delimiters to Obsidian format ($...$ and $$...$$):
    pdf2md --fix-math <path-to-file-or-dir>

    # With backend flag:
    pdf2md --backend <backend> <input-path> <output-path> [begin-page] [end-page]

Supported backends:
    qwen   - Locally deployed Qwen Coder (default, via Ollama/vLLM/OpenAI-compatible endpoint)
    gemini - Google Gemini API (requires GEMINI_API_KEY)
    gemma  - Google Gemma API (requires GEMMA_API_KEY or GEMINI_API_KEY)

Environment variables:
    PDF2MD_BACKEND      Default OCR backend (default: qwen)
    PDF2MD_CONCURRENCY  Number of concurrent page workers (default: 2)
    QWEN_URL            Local Qwen endpoint URL (default: http://localhost:11434/v1)
    QWEN_MODEL          Model name (default: qwen2.5-coder:latest)
    QWEN_API_KEY        Optional API key for local endpoint
    GEMINI_API_KEY      Gemini API key
    GEMMA_API_KEY       Gemma API key
"#
    );
}

/// Main entrypoint for CLI command execution.
pub fn run() {
    let raw_args: Vec<String> = env::args().skip(1).collect();

    if raw_args.is_empty() || raw_args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage();
        process::exit(if raw_args.is_empty() { 1 } else { 0 });
    }

    if raw_args[0] == "--fix-math" || raw_args[0] == "--normalize-math" {
        if raw_args.len() < 2 {
            eprintln!(
                "Error: Flag '{}' requires a file or directory path",
                raw_args[0]
            );
            process::exit(1);
        }
        let target = Path::new(&raw_args[1]);
        if !target.exists() {
            eprintln!("Error: Path '{}' does not exist", target.display());
            process::exit(1);
        }
        match fix_math_in_path(target) {
            Ok(count) => {
                println!(
                    "Successfully normalized LaTeX math delimiters in {count} markdown file(s)."
                );
                process::exit(0);
            }
            Err(e) => {
                eprintln!("Error normalizing math: {e:#}");
                process::exit(1);
            }
        }
    }

    let mut backend_override: Option<OcrBackend> = None;
    let mut positional_args: Vec<String> = Vec::new();

    let mut i = 0;
    while i < raw_args.len() {
        let arg = &raw_args[i];
        if arg == "--backend" || arg == "-b" {
            if i + 1 >= raw_args.len() {
                eprintln!(
                    "Error: Flag '{}' requires a backend argument (qwen, gemini, gemma)",
                    arg
                );
                print_usage();
                process::exit(1);
            }
            backend_override = match raw_args[i + 1].parse::<OcrBackend>() {
                Ok(b) => Some(b),
                Err(err) => {
                    eprintln!("Error: {err}");
                    process::exit(1);
                }
            };
            i += 2;
        } else if let Some(val) = arg.strip_prefix("--backend=") {
            backend_override = match val.parse::<OcrBackend>() {
                Ok(b) => Some(b),
                Err(err) => {
                    eprintln!("Error: {err}");
                    process::exit(1);
                }
            };
            i += 1;
        } else {
            positional_args.push(arg.clone());
            i += 1;
        }
    }

    if positional_args.len() < 2 || positional_args.len() > 5 {
        print_usage();
        process::exit(1);
    }

    let input_path = Path::new(&positional_args[0]);
    let output_path = Path::new(&positional_args[1]);

    let (page_range, positional_backend) = match positional_args.len() {
        2 => (None, None),
        3 => match positional_args[2].parse::<OcrBackend>() {
            Ok(b) => (None, Some(b)),
            Err(_) => {
                eprintln!(
                    "Error: 3 arguments provided, but '{}' is not a recognized backend (expected: qwen, gemini, gemma)",
                    positional_args[2]
                );
                print_usage();
                process::exit(1);
            }
        },
        4 => {
            let begin_page = match positional_args[2].parse::<usize>() {
                Ok(page) => page,
                Err(_) => {
                    eprintln!(
                        "Invalid begin page '{}': expected a positive integer",
                        positional_args[2]
                    );
                    process::exit(1);
                }
            };
            let end_page = match positional_args[3].parse::<usize>() {
                Ok(page) => page,
                Err(_) => {
                    eprintln!(
                        "Invalid end page '{}': expected a positive integer",
                        positional_args[3]
                    );
                    process::exit(1);
                }
            };
            (Some((begin_page, end_page)), None)
        }
        5 => {
            let begin_page = match positional_args[2].parse::<usize>() {
                Ok(page) => page,
                Err(_) => {
                    eprintln!(
                        "Invalid begin page '{}': expected a positive integer",
                        positional_args[2]
                    );
                    process::exit(1);
                }
            };
            let end_page = match positional_args[3].parse::<usize>() {
                Ok(page) => page,
                Err(_) => {
                    eprintln!(
                        "Invalid end page '{}': expected a positive integer",
                        positional_args[3]
                    );
                    process::exit(1);
                }
            };
            let b = match positional_args[4].parse::<OcrBackend>() {
                Ok(b) => b,
                Err(err) => {
                    eprintln!("Error: {err}");
                    process::exit(1);
                }
            };
            (Some((begin_page, end_page)), Some(b))
        }
        _ => unreachable!(),
    };

    let backend = if let Some(b) = backend_override {
        b
    } else if let Some(b) = positional_backend {
        b
    } else {
        resolve_default_backend()
    };

    if !input_path.exists() {
        eprintln!(
            "Error: Input path '{}' does not exist.",
            input_path.display()
        );
        process::exit(1);
    }

    if input_path.is_dir() {
        match process_pdf_directory(input_path, output_path, page_range, backend) {
            Ok(summary) => {
                if !summary.failed.is_empty() {
                    process::exit(1);
                }
            }
            Err(err) => {
                eprintln!("Batch PDF conversion failed: {err:#}");
                process::exit(1);
            }
        }
    } else {
        let (begin_page, end_page) = match page_range {
            Some((b, e)) => (b, e),
            None => {
                let total = match get_pdf_page_count(input_path) {
                    Ok(t) => t,
                    Err(err) => {
                        eprintln!(
                            "Error determining total pages for {}: {err:#}",
                            input_path.display()
                        );
                        process::exit(1);
                    }
                };
                (1, total)
            }
        };

        let is_output_dir = output_path.is_dir()
            || output_path
                .extension()
                .is_none_or(|ext| !ext.eq_ignore_ascii_case("md"));

        let final_output_md = if is_output_dir {
            let stem = input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("document");
            let clean_name = sanitize_folder_name(stem);
            let unique_dir = output_path.join(&clean_name);
            if let Err(err) = fs::create_dir_all(&unique_dir) {
                eprintln!(
                    "Failed to create output directory {}: {err:#}",
                    unique_dir.display()
                );
                process::exit(1);
            }
            unique_dir.join(format!("{clean_name}.md"))
        } else {
            output_path.to_path_buf()
        };

        if let Err(err) =
            process_pdf_with_backend(input_path, &final_output_md, begin_page, end_page, backend)
        {
            eprintln!("PDF conversion failed: {err:#}");
            process::exit(1);
        }
    }
}

pub fn fix_math_in_path(path: &Path) -> anyhow::Result<usize> {
    let mut count = 0;
    if path.is_file() {
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
            && fix_math_in_file(path)?
        {
            count += 1;
        }
    } else if path.is_dir() {
        count = fix_math_in_dir(path)?;
    }
    Ok(count)
}

fn fix_math_in_dir(dir: &Path) -> anyhow::Result<usize> {
    let mut count = 0;
    let entries = fs::read_dir(dir)?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            count += fix_math_in_dir(&p)?;
        } else if p.is_file()
            && p.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
            && fix_math_in_file(&p)?
        {
            count += 1;
        }
    }
    Ok(count)
}

fn fix_math_in_file(file: &Path) -> anyhow::Result<bool> {
    let content = fs::read_to_string(file)?;
    let normalized = crate::markdown::normalize_latex_delimiters(&content);
    if normalized != content {
        fs::write(file, normalized)?;
        println!("✓ Normalized math formatting: {}", file.display());
        Ok(true)
    } else {
        Ok(false)
    }
}
