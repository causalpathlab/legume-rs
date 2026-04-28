//! Qualitative color palettes for `senna plot`.
//!
//! Default picks are driven by topic count, favoring colorblind-safe
//! palettes below 8 and falling back to Tableau Category10/20 for larger
//! palettes. Hardcoded RGB tables (rather than pulling a palette library
//! at runtime) keep output reproducible and spec-aligned to the published
//! palette definitions.
//!
//! ## References
//! - Brewer, C. A. *ColorBrewer 2.0.* <https://colorbrewer2.org/>
//! - Okabe, M., & Ito, K. (2008). *Color Universal Design (CUD): How to
//!   make figures and presentations that are friendly to colorblind
//!   people.* <https://jfly.uni-koeln.de/color/>
//! - Tableau Category10 / Category20 via Mike Bostock's
//!   d3-scale-chromatic.

use clap::ValueEnum;

/// `(r, g, b)` in 0..=255.
pub type Rgb = (u8, u8, u8);

/// Qualitative palette selection. `Auto` picks by topic count.
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
pub enum Palette {
    /// Auto: Paired for ≤11 (default), Category20 beyond. Yellow is
    /// stripped from every palette in this module — illegible on white.
    Auto,
    /// Okabe–Ito 8-color colorblind-safe palette.
    OkabeIto,
    /// ColorBrewer Dark2 (≤8).
    Dark2,
    /// ColorBrewer Set1 (≤9).
    Set1,
    /// ColorBrewer Set2 (≤8).
    Set2,
    /// ColorBrewer Set3 (≤12).
    Set3,
    /// ColorBrewer Paired (≤12).
    Paired,
    /// Tableau Category10 (≤10).
    Category10,
    /// Tableau-esque Category20 (≤20).
    Category20,
}

// Okabe-Ito's canonical first stop is black; we drop it because pinto
// reserves black for the interfaces overlay and uses qualitative palettes
// for community/topic IDs that should never collide with that signal.
// We also drop pure yellow (240, 228, 66) — illegible on white slides.
const OKABE_ITO: &[Rgb] = &[
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (0, 114, 178),
    (213, 94, 0),
    (204, 121, 167),
];

// Yellow (230, 171, 2) dropped from Dark2.
const DARK2: &[Rgb] = &[
    (27, 158, 119),
    (217, 95, 2),
    (117, 112, 179),
    (231, 41, 138),
    (102, 166, 30),
    (166, 118, 29),
    (102, 102, 102),
];

// Yellow (255, 255, 51) dropped from Set1.
const SET1: &[Rgb] = &[
    (228, 26, 28),
    (55, 126, 184),
    (77, 175, 74),
    (152, 78, 163),
    (255, 127, 0),
    (166, 86, 40),
    (247, 129, 191),
    (153, 153, 153),
];

// Yellow (255, 217, 47) dropped from Set2.
const SET2: &[Rgb] = &[
    (102, 194, 165),
    (252, 141, 98),
    (141, 160, 203),
    (231, 138, 195),
    (166, 216, 84),
    (229, 196, 148),
    (179, 179, 179),
];

// Pale yellows (255, 255, 179) and (255, 237, 111) dropped from Set3.
const SET3: &[Rgb] = &[
    (141, 211, 199),
    (190, 186, 218),
    (251, 128, 114),
    (128, 177, 211),
    (253, 180, 98),
    (179, 222, 105),
    (252, 205, 229),
    (217, 217, 217),
    (188, 128, 189),
    (204, 235, 197),
];

// Pale yellow (255, 255, 153) dropped from Paired.
const PAIRED: &[Rgb] = &[
    (166, 206, 227),
    (31, 120, 180),
    (178, 223, 138),
    (51, 160, 44),
    (251, 154, 153),
    (227, 26, 28),
    (253, 191, 111),
    (255, 127, 0),
    (202, 178, 214),
    (106, 61, 154),
    (177, 89, 40),
];

// Olive-yellow (188, 189, 34) dropped from Category10.
const CATEGORY10: &[Rgb] = &[
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (23, 190, 207),
];

// Olive-yellows (188, 189, 34) and (219, 219, 141) dropped from Category20.
const CATEGORY20: &[Rgb] = &[
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (23, 190, 207),
    (158, 218, 229),
];

/// Resolve `Auto` to a concrete palette based on `n` topics.
#[must_use]
pub fn resolve(palette: &Palette, n: usize) -> Palette {
    if *palette != Palette::Auto {
        return palette.clone();
    }
    // ColorBrewer Paired (yellow dropped) is the default — gives 11
    // distinct paired light/dark hues that read cleanly on slides and
    // print. Beyond that, Category20 (also yellow-stripped) keeps the
    // distinguishability up to ~18 communities; further out we cycle
    // Category20.
    if n <= PAIRED.len() {
        Palette::Paired
    } else {
        Palette::Category20
    }
}

/// Return the `i`-th color of the palette. Cycles if `i >= size`.
/// `Auto` falls through to `Paired` here as its base table — most
/// callers use [`resolve`] first, but this fallback keeps direct
/// `color(&Palette::Auto, i)` calls sane.
#[must_use]
pub fn color(palette: &Palette, i: usize) -> Rgb {
    let table: &[Rgb] = match palette {
        Palette::Auto | Palette::Paired => PAIRED,
        Palette::OkabeIto => OKABE_ITO,
        Palette::Dark2 => DARK2,
        Palette::Set1 => SET1,
        Palette::Set2 => SET2,
        Palette::Set3 => SET3,
        Palette::Category10 => CATEGORY10,
        Palette::Category20 => CATEGORY20,
    };
    table[i % table.len()]
}
