use anyhow::{Context, Result, anyhow};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::backends::OcrBackend;
use crate::pdf::get_pdf_page_count;
use crate::pipeline::process_pdf_with_backend;

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

/// Default directory search depth for PDF batch conversion (searches root and up to 3 nested subdirectory levels).
pub const DEFAULT_MAX_SEARCH_DEPTH: usize = 3;

/// Resolves maximum directory depth for recursive PDF search from `PDF2MD_MAX_DEPTH` (default: 3).
pub fn resolve_max_search_depth() -> usize {
    if let Ok(val) = std::env::var("PDF2MD_MAX_DEPTH")
        && let Ok(parsed) = val.parse::<usize>()
    {
        return parsed;
    }
    DEFAULT_MAX_SEARCH_DEPTH
}

/// Collects all PDF files (`.pdf`, case-insensitive) in a directory searching up to `max_depth` directory levels, sorted by name.
pub fn collect_pdf_files_with_depth(dir: &Path, max_depth: usize) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Err(anyhow!("Directory '{}' does not exist", dir.display()));
    }
    let mut pdfs = Vec::new();
    collect_pdf_files_recursive(dir, 0, max_depth, &mut pdfs)?;
    pdfs.sort();
    Ok(pdfs)
}

/// Collects all PDF files (`.pdf`, case-insensitive) in a directory searching at least 3 levels of directories, sorted by name.
pub fn collect_pdf_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let max_depth = resolve_max_search_depth();
    collect_pdf_files_with_depth(dir, max_depth)
}

fn collect_pdf_files_recursive(
    current_dir: &Path,
    current_depth: usize,
    max_depth: usize,
    collected: &mut Vec<PathBuf>,
) -> Result<()> {
    let entries = match fs::read_dir(current_dir) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!(
                "Warning: could not read directory '{}': {err}",
                current_dir.display()
            );
            return Ok(());
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };

        if file_type.is_file() {
            if let Some(ext) = path.extension()
                && ext.eq_ignore_ascii_case("pdf")
            {
                collected.push(path);
            }
        } else if file_type.is_dir() && current_depth < max_depth {
            // Ignore hidden directories (e.g., .git, .obsidian, .trash)
            let is_hidden = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|name| name.starts_with('.'));

            if !is_hidden {
                collect_pdf_files_recursive(&path, current_depth + 1, max_depth, collected)?;
            }
        }
    }

    Ok(())
}

/// Finds a unique folder path inside `base_dir` for the given `folder_name`.
pub fn resolve_unique_output_dir(base_dir: &Path, folder_name: &str) -> PathBuf {
    let candidate = base_dir.join(folder_name);
    if !candidate.exists() {
        return candidate;
    }
    // If it exists and is an empty directory, reuse it
    if candidate.is_dir()
        && let Ok(mut entries) = fs::read_dir(&candidate)
        && entries.next().is_none()
    {
        return candidate;
    }
    let mut counter = 1;
    loop {
        let next = base_dir.join(format!("{folder_name}_{counter}"));
        if !next.exists() {
            return next;
        }
        if next.is_dir()
            && let Ok(mut entries) = fs::read_dir(&next)
            && entries.next().is_none()
        {
            return next;
        }
        counter += 1;
    }
}

/// Summary report for batch conversion runs.
#[derive(Debug, Default, Clone)]
pub struct BatchConversionResult {
    pub total: usize,
    pub succeeded: usize,
    pub failed: Vec<(PathBuf, String)>,
}

/// Converts all PDF files in `input_dir` one by one, outputting each into a uniquely named folder in `output_base_dir`.
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

    let mut used_folder_names = HashSet::new();

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

        let total_pages = match get_pdf_page_count(pdf_path) {
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
            pdf_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("document.pdf"),
            begin_page,
            end_page,
            total_pages,
            pdf_output_dir.display(),
            target_md_path.display(),
            pdf_output_dir
                .join(format!(
                    "{}_images",
                    pdf_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or(&folder_name)
                ))
                .display()
        );

        match process_pdf_with_backend(pdf_path, &target_md_path, begin_page, end_page, backend) {
            Ok(()) => {
                result.succeeded += 1;
                println!(
                    "✓ Successfully converted '{}' -> {}",
                    pdf_path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("document.pdf"),
                    target_md_path.display()
                );
            }
            Err(err) => {
                eprintln!("✗ Failed to convert '{}': {err:#}", pdf_path.display());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_folder_name() {
        assert_eq!(sanitize_folder_name("CC_code"), "CC_code");
        assert_eq!(sanitize_folder_name("TB_26-04 FInal"), "TB_26-04 FInal");
        assert_eq!(sanitize_folder_name("doc:name*test?"), "doc_name_test_");
        assert_eq!(sanitize_folder_name("   "), "document");
    }

    #[test]
    fn test_resolve_unique_output_dir() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path();

        let dir1 = resolve_unique_output_dir(base, "doc");
        assert_eq!(dir1, base.join("doc"));
        fs::create_dir_all(&dir1).unwrap();
        // Create a file inside dir1 so it's not empty
        fs::write(dir1.join("file.txt"), "hello").unwrap();

        let dir2 = resolve_unique_output_dir(base, "doc");
        assert_eq!(dir2, base.join("doc_1"));
    }

    #[test]
    fn test_collect_pdf_files() {
        let p = Path::new("input/CEA");
        if p.exists() {
            let files = collect_pdf_files(p).unwrap();
            assert!(!files.is_empty());
            assert!(files.iter().all(|f| {
                f.extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
            }));
        }
    }

    #[test]
    fn test_collect_pdf_files_recursive_three_levels() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();

        // Level 0
        fs::write(root.join("root.pdf"), b"%PDF-1.4").unwrap();
        fs::write(root.join("ignored.txt"), b"not a pdf").unwrap();

        // Level 1
        let l1 = root.join("level1");
        fs::create_dir_all(&l1).unwrap();
        fs::write(l1.join("doc1.pdf"), b"%PDF-1.4").unwrap();

        // Level 2
        let l2 = l1.join("level2");
        fs::create_dir_all(&l2).unwrap();
        fs::write(l2.join("doc2.pdf"), b"%PDF-1.4").unwrap();

        // Level 3
        let l3 = l2.join("level3");
        fs::create_dir_all(&l3).unwrap();
        fs::write(l3.join("doc3.PDF"), b"%PDF-1.4").unwrap();

        // Level 4 (should be excluded when max_depth is 3)
        let l4 = l3.join("level4");
        fs::create_dir_all(&l4).unwrap();
        fs::write(l4.join("doc4.pdf"), b"%PDF-1.4").unwrap();

        // Hidden directory at level 1 (should be excluded)
        let hidden = root.join(".git");
        fs::create_dir_all(&hidden).unwrap();
        fs::write(hidden.join("hidden.pdf"), b"%PDF-1.4").unwrap();

        // Test default depth (3 levels)
        let files_default = collect_pdf_files(root).unwrap();
        let file_names: Vec<String> = files_default
            .iter()
            .map(|f| f.file_name().unwrap().to_str().unwrap().to_string())
            .collect();

        assert!(file_names.contains(&"root.pdf".to_string()));
        assert!(file_names.contains(&"doc1.pdf".to_string()));
        assert!(file_names.contains(&"doc2.pdf".to_string()));
        assert!(file_names.contains(&"doc3.PDF".to_string()));
        assert!(!file_names.contains(&"doc4.pdf".to_string()));
        assert!(!file_names.contains(&"hidden.pdf".to_string()));
        assert!(!file_names.contains(&"ignored.txt".to_string()));
        assert_eq!(files_default.len(), 4);

        // Test custom depth 4 (should include level 4)
        let files_depth4 = collect_pdf_files_with_depth(root, 4).unwrap();
        assert_eq!(files_depth4.len(), 5);

        // Test custom depth 1 (only root and level 1)
        let files_depth1 = collect_pdf_files_with_depth(root, 1).unwrap();
        assert_eq!(files_depth1.len(), 2);
    }
}
