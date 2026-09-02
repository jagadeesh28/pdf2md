use anyhow::{Context, Result, anyhow};
use hayro::hayro_interpret::InterpreterSettings;
use hayro::hayro_syntax::Pdf;
use std::fs;
use std::path::{Path, PathBuf};

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
    println!("REached 1");
    let document = Pdf::new(pdf_bytes).map_err(|error| {
        anyhow!(
            "failed to parse PDF for rendering {}: {error:?}",
            pdf_path.display()
        )
    })?;
    let cache = hayro::RenderCache::new();
    let interpreter_settings = InterpreterSettings::default();
    let render_settings = hayro::RenderSettings {
        x_scale: 2.0,
        y_scale: 2.0,
        bg_color: hayro::vello_cpu::color::palette::css::WHITE,
        ..Default::default()
    };
    
    let mut page_paths = Vec::new();
    if end_page > document.pages().len() {
        return Err(anyhow!(
            "page range {}-{} is outside the PDF's {} pages",
            begin_page,
            end_page,
            document.pages().len()
        ));
    }

    for (index, page) in document
        .pages()
        .iter()
        .enumerate()
        .skip(begin_page - 1)
        .take(end_page - begin_page + 1)
    {
        let page_path = output_dir.join(format!("page-{}.png", index + 1));
        let png = hayro::render(page, &cache, &interpreter_settings, &render_settings)
            .into_png()
            .with_context(|| format!("failed to encode rendered PDF page {}", index + 1))?;
        fs::write(&page_path, png)
            .with_context(|| format!("failed to save rendered page {}", index + 1))?;
        page_paths.push(page_path);
    }

    Ok(page_paths)
}
