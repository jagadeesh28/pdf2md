use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, Luma};
use std::path::Path;

/// Summed-area tables (integral images) for O(1) local mean and variance queries.
pub struct IntegralImages {
    pub width: usize,
    pub sum: Vec<u64>,
    pub sum_sq: Vec<u64>,
}

impl IntegralImages {
    /// Compute integral image and squared integral image in a single O(W * H) pass.
    pub fn from_gray(gray: &GrayImage) -> Self {
        let (w, h) = gray.dimensions();
        let w_usize = w as usize;
        let h_usize = h as usize;
        let stride = w_usize + 1;
        let total = stride * (h_usize + 1);

        let mut sum = vec![0u64; total];
        let mut sum_sq = vec![0u64; total];

        for y in 0..h_usize {
            let mut row_sum = 0u64;
            let mut row_sum_sq = 0u64;
            let py = y + 1;
            let prev_row_start = y * stride;
            let curr_row_start = py * stride;

            for x in 0..w_usize {
                let px = x + 1;
                let val = gray.get_pixel(x as u32, y as u32)[0] as u64;
                row_sum += val;
                row_sum_sq += val * val;

                sum[curr_row_start + px] = sum[prev_row_start + px] + row_sum;
                sum_sq[curr_row_start + px] = sum_sq[prev_row_start + px] + row_sum_sq;
            }
        }

        Self {
            width: w_usize,
            sum,
            sum_sq,
        }
    }

    /// Query the sum and squared sum over the bounding box [x0, x1] x [y0, y1] inclusive in O(1).
    #[inline]
    pub fn query(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> (u64, u64, usize) {
        let stride = self.width + 1;
        let area = (x1 - x0 + 1) * (y1 - y0 + 1);

        let br = (y1 + 1) * stride + (x1 + 1);
        let tr = y0 * stride + (x1 + 1);
        let bl = (y1 + 1) * stride + x0;
        let tl = y0 * stride + x0;

        let s = self.sum[br] + self.sum[tl] - self.sum[tr] - self.sum[bl];
        let s_sq = self.sum_sq[br] + self.sum_sq[tl] - self.sum_sq[tr] - self.sum_sq[bl];

        (s, s_sq, area)
    }
}

/// Image preprocessor providing adaptive Sauvola binarization and morphological despeckle filter.
pub struct ImagePreprocessor;

impl ImagePreprocessor {
    /// Perform high-performance Sauvola adaptive thresholding in O(W * H) time using integral images.
    /// Converts noisy scanned grayscale figures into crisp 1-bit monochrome line art.
    ///
    /// Formula:
    ///   T(x, y) = m(x, y) * [1 + k * (s(x, y) / R - 1)]
    ///   where:
    ///     m(x, y) = local mean across window
    ///     s(x, y) = local standard deviation across window
    ///     R = dynamic range (128.0 for 8-bit images)
    ///     k = Sauvola parameter (typically 0.1 to 0.3, default 0.2)
    pub fn sauvola_binarize(img: &DynamicImage, window_size: u32, k: f32) -> GrayImage {
        let gray = img.to_luma8();
        let (width, height) = gray.dimensions();
        if width == 0 || height == 0 {
            return gray;
        }

        let integral = IntegralImages::from_gray(&gray);
        let mut binarized = GrayImage::new(width, height);
        let half_win = (window_size / 2) as usize;
        let r = 128.0f32;
        let w_max = width as usize - 1;
        let h_max = height as usize - 1;

        for y in 0..height as usize {
            let y_min = y.saturating_sub(half_win);
            let y_max = (y + half_win).min(h_max);

            for x in 0..width as usize {
                let x_min = x.saturating_sub(half_win);
                let x_max = (x + half_win).min(w_max);

                let (sum, sum_sq, area) = integral.query(x_min, y_min, x_max, y_max);
                let count_f = area as f32;
                let mean = sum as f32 / count_f;
                let variance = (sum_sq as f32 / count_f) - (mean * mean);
                let std_dev = variance.max(0.0).sqrt();

                let threshold = mean * (1.0 + k * (std_dev / r - 1.0));
                let pixel_val = gray.get_pixel(x as u32, y as u32)[0] as f32;

                if pixel_val > threshold {
                    binarized.put_pixel(x as u32, y as u32, Luma([255])); // Pure white background
                } else {
                    binarized.put_pixel(x as u32, y as u32, Luma([0])); // Pure black stroke
                }
            }
        }

        binarized
    }

    /// Morphological despeckle filter that removes scanner dust (isolated 1-2px black noise)
    /// and fills pinholes (isolated 1-2px white holes in black strokes) while strictly
    /// preserving thin mechanical lines (>= 3px), corners, and alphanumeric characters.
    pub fn despeckle(img: &GrayImage, iterations: u32) -> GrayImage {
        let (w, h) = img.dimensions();
        if w < 3 || h < 3 || iterations == 0 {
            return img.clone();
        }

        let mut current = img.clone();
        let mut next = img.clone();

        for _ in 0..iterations {
            for y in 0..h {
                for x in 0..w {
                    let pixel = current.get_pixel(x, y)[0];

                    if pixel == 0 {
                        // Black pixel: check if it's isolated noise (pepper)
                        let (count, single) =
                            Self::count_matching_neighbors(&current, x, y, w, h, 0);
                        if count == 0 {
                            // Completely isolated 1-pixel black noise -> flip to white
                            next.put_pixel(x, y, Luma([255]));
                        } else if count == 1 {
                            // Check if the only neighbor also has no other black neighbors
                            if let Some((nx, ny)) = single {
                                let (n_count, n_single) =
                                    Self::count_matching_neighbors(&current, nx, ny, w, h, 0);
                                if n_count == 1 && n_single == Some((x, y)) {
                                    // Isolated 2-pixel cluster surrounded by white -> flip to white
                                    next.put_pixel(x, y, Luma([255]));
                                }
                            }
                        }
                    } else {
                        // White pixel: check if it's an isolated pinhole in solid black (salt)
                        let (white_count, white_single) =
                            Self::count_matching_neighbors(&current, x, y, w, h, 255);
                        if white_count == 0 {
                            // Completely enclosed 1-pixel white hole inside black stroke -> flip to black
                            next.put_pixel(x, y, Luma([0]));
                        } else if white_count == 1 {
                            if let Some((nx, ny)) = white_single {
                                let (n_white_count, n_white_single) =
                                    Self::count_matching_neighbors(&current, nx, ny, w, h, 255);
                                if n_white_count == 1 && n_white_single == Some((x, y)) {
                                    // Isolated 2-pixel white hole inside black stroke -> flip to black
                                    next.put_pixel(x, y, Luma([0]));
                                }
                            }
                        }
                    }
                }
            }
            current = next.clone();
        }

        current
    }

    /// Zero-allocation helper to count matching 8-connected neighbors and return the first neighbor if single.
    #[inline(always)]
    fn count_matching_neighbors(
        img: &GrayImage,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        target_val: u8,
    ) -> (u8, Option<(u32, u32)>) {
        let mut count = 0u8;
        let mut first = None;
        let x_i = x as i32;
        let y_i = y as i32;

        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x_i + dx;
                let ny = y_i + dy;
                if nx >= 0
                    && nx < w as i32
                    && ny >= 0
                    && ny < h as i32
                    && img.get_pixel(nx as u32, ny as u32)[0] == target_val
                {
                    count += 1;
                    if count == 1 {
                        first = Some((nx as u32, ny as u32));
                    }
                }
            }
        }

        (count, first)
    }

    /// Enhances a cropped figure file on disk in place:
    /// 1. Reads image from `image_path`
    /// 2. Applies Sauvola adaptive thresholding with dynamically scaled window size
    /// 3. Applies morphological despeckle filter
    /// 4. Saves enhanced image back to `image_path`
    pub fn enhance_image_file(image_path: &Path) -> Result<()> {
        let raw_img = image::open(image_path)
            .with_context(|| format!("failed to open image at {}", image_path.display()))?;

        let (w, h) = (raw_img.width(), raw_img.height());
        if w < 10 || h < 10 {
            return Ok(());
        }

        // Compute window size proportional to image resolution (odd integer between 15 and 51)
        let win = ((w.min(h) / 10).clamp(15, 51) | 1) as u32;
        let binarized = Self::sauvola_binarize(&raw_img, win, 0.2);
        let despeckled = Self::despeckle(&binarized, 2);

        despeckled
            .save(image_path)
            .with_context(|| format!("failed to save enhanced image to {}", image_path.display()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sauvola_synthetic_contrast() {
        let mut img = GrayImage::new(50, 50);
        for y in 0..50 {
            for x in 0..50 {
                let gradient_bg = (200 + (x + y) / 2).min(255) as u8;
                img.put_pixel(x, y, Luma([gradient_bg]));
            }
        }

        // Draw a dark square stroke in the center
        for y in 20..30 {
            for x in 20..30 {
                img.put_pixel(x, y, Luma([40]));
            }
        }

        let dyn_img = DynamicImage::ImageLuma8(img);
        let binarized = ImagePreprocessor::sauvola_binarize(&dyn_img, 15, 0.2);

        // Check stroke is black (0) and background is white (255)
        assert_eq!(binarized.get_pixel(25, 25)[0], 0);
        assert_eq!(binarized.get_pixel(10, 10)[0], 255);
        assert_eq!(binarized.get_pixel(40, 40)[0], 255);
    }

    #[test]
    fn test_despeckle_removes_noise_preserves_lines() {
        let mut img = GrayImage::new(20, 20);
        for y in 0..20 {
            for x in 0..20 {
                img.put_pixel(x, y, Luma([255]));
            }
        }

        // 1-pixel noise at (2, 2)
        img.put_pixel(2, 2, Luma([0]));

        // 2-pixel noise at (15, 2) and (16, 2)
        img.put_pixel(15, 2, Luma([0]));
        img.put_pixel(16, 2, Luma([0]));

        // 4-pixel line from (5, 10) to (8, 10)
        img.put_pixel(5, 10, Luma([0]));
        img.put_pixel(6, 10, Luma([0]));
        img.put_pixel(7, 10, Luma([0]));
        img.put_pixel(8, 10, Luma([0]));

        let cleaned = ImagePreprocessor::despeckle(&img, 1);

        assert_eq!(cleaned.get_pixel(2, 2)[0], 255, "1px noise should be removed");
        assert_eq!(cleaned.get_pixel(15, 2)[0], 255, "2px noise should be removed");
        assert_eq!(cleaned.get_pixel(16, 2)[0], 255, "2px noise should be removed");

        // Line must be preserved
        assert_eq!(cleaned.get_pixel(5, 10)[0], 0);
        assert_eq!(cleaned.get_pixel(6, 10)[0], 0);
        assert_eq!(cleaned.get_pixel(7, 10)[0], 0);
        assert_eq!(cleaned.get_pixel(8, 10)[0], 0);
    }
}
