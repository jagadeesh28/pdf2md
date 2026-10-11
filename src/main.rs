use pdf2md::OcrBackend;
use std::env;
use std::path::Path;
use std::process;

fn print_usage() {
    eprintln!(
        r#"Usage:
    pdf2md <input.pdf> <output.md> <begin-page> <end-page> [backend]
    pdf2md --backend <backend> <input.pdf> <output.md> <begin-page> <end-page>

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

fn main() {
    let raw_args: Vec<String> = env::args().skip(1).collect();

    if raw_args.is_empty() || raw_args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage();
        process::exit(if raw_args.is_empty() { 1 } else { 0 });
    }

    let mut backend_override: Option<OcrBackend> = None;
    let mut positional_args: Vec<String> = Vec::new();

    let mut i = 0;
    while i < raw_args.len() {
        let arg = &raw_args[i];
        if arg == "--backend" || arg == "-b" {
            if i + 1 >= raw_args.len() {
                eprintln!("Error: Flag '{}' requires a backend argument (qwen, gemini, gemma)", arg);
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

    if positional_args.len() != 4 && positional_args.len() != 5 {
        print_usage();
        process::exit(1);
    }

    let input_path = Path::new(&positional_args[0]);
    let output_path = Path::new(&positional_args[1]);
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

    let backend = if let Some(b) = backend_override {
        b
    } else if positional_args.len() == 5 {
        match positional_args[4].parse::<OcrBackend>() {
            Ok(b) => b,
            Err(err) => {
                eprintln!("Error: {err}");
                process::exit(1);
            }
        }
    } else {
        pdf2md::resolve_default_backend()
    };

    if let Err(err) = pdf2md::process_pdf_with_backend(
        input_path,
        output_path,
        begin_page,
        end_page,
        backend,
    ) {
        eprintln!("PDF conversion failed: {err:#}");
        process::exit(1);
    }
}
