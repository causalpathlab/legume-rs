use genomic_data::bed::Bed;
use genomic_data::sam::Strand;
use std::io::BufRead;

/// A 3'-UTR region parsed from a BED file.
pub struct UtrRegion {
    pub chr: Box<str>,
    pub start: i64,
    pub end: i64,
    pub strand: Strand,
    pub name: Box<str>,
    pub utr_length: usize,
}

impl UtrRegion {
    pub fn to_bed(&self) -> Bed {
        Bed {
            chr: self.chr.clone(),
            start: self.start,
            stop: self.end,
        }
    }

    /// Convert a UTR-relative alpha position to a genomic range [start, stop].
    /// For + strand: genomic_alpha = utr.start + alpha
    /// For - strand: genomic_alpha = utr.end - alpha
    /// Range: [genomic_alpha - beta, genomic_alpha + beta]
    pub fn alpha_to_genomic_range(&self, alpha: f64, beta: f64) -> (i64, i64) {
        let genomic_alpha = match self.strand {
            Strand::Forward => self.start + alpha as i64,
            Strand::Backward => self.end - alpha as i64,
        };
        let start = (genomic_alpha - beta as i64).max(0);
        let stop = genomic_alpha + beta as i64;
        (start, stop)
    }
}

/// Load UTR regions from a BED file.
///
/// Supports multiple formats:
/// - 4-col SCAPE format: chr, start, end, strand (+/-)
/// - 6-col standard BED: chr, start, end, name, score, strand
/// - 3-col minimal: chr, start, end (assumes forward strand)
pub fn load_utr_regions_from_bed(path: &str) -> anyhow::Result<Vec<UtrRegion>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut regions = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            continue;
        }

        let chr: Box<str> = fields[0].into();
        let start: i64 = fields[1].parse()?;
        let end: i64 = fields[2].parse()?;

        // Detect format based on column count and content
        let (name, strand) = if fields.len() >= 6 {
            // Standard 6-col BED: chr, start, end, name, score, strand
            let name: Box<str> = fields[3].into();
            let strand = match fields[5] {
                "-" => Strand::Backward,
                _ => Strand::Forward,
            };
            (name, strand)
        } else if fields.len() == 4 {
            // Could be SCAPE format (chr, start, end, strand) or (chr, start, end, name)
            let col3 = fields[3];
            if col3 == "+" || col3 == "-" {
                // SCAPE 4-col format: strand in column 4
                let strand = if col3 == "-" {
                    Strand::Backward
                } else {
                    Strand::Forward
                };
                let name: Box<str> = format!("{}:{}-{}", chr, start, end).into();
                (name, strand)
            } else {
                // name in column 4, assume forward strand
                (col3.into(), Strand::Forward)
            }
        } else {
            // 3-col: chr, start, end
            let name: Box<str> = format!("{}:{}-{}", chr, start, end).into();
            (name, Strand::Forward)
        };

        let utr_length = (end - start) as usize;

        regions.push(UtrRegion {
            chr,
            start,
            end,
            strand,
            name,
            utr_length,
        });
    }

    Ok(regions)
}
