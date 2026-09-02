use std::env;
use std::path::Path;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 5 {
        eprintln!("Usage: pdf2md <input.pdf> <output.md> <begin-page> <end-page>");
        process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);
    let begin_page = match args[3].parse::<usize>() {
        Ok(page) => page,
        Err(_) => {
            eprintln!("Invalid begin page '{}': expected a positive integer", args[3]);
            process::exit(1);
        }
    };
    let end_page = match args[4].parse::<usize>() {
        Ok(page) => page,
        Err(_) => {
            eprintln!("Invalid end page '{}': expected a positive integer", args[4]);
            process::exit(1);
        }
    };

    if let Err(err) = pdf2md::process_pdf(input_path, output_path, begin_page, end_page) {
        eprintln!("PDF conversion failed: {err:#}");
        process::exit(1);
    }
}
