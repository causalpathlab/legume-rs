pub mod scatter;
pub mod strand;
pub mod topic;

pub use plot_utils::{hull, palette, rasterize, svg_emit};

/// Map a label (cell type, batch, …) to a filesystem-safe basename:
/// keep ASCII alphanumerics and `-_.`, replace everything else with `_`.
pub(crate) fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | '.' => c,
            _ => '_',
        })
        .collect()
}
