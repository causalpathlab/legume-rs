use crate::dartseq_sifter::MethylatedSite;
use crate::data::dna::Dna;
use crate::data::gff::*;
use crate::data::sam::Strand;
use dashmap::DashMap as HashMap;
use matrix_util::common_io;
use std::io::Write;

/// Represents a bin of methylation statistics
#[derive(Eq, Hash, PartialEq, PartialOrd, Ord, Clone)]
pub struct MethBin {
    neg_log10_p_wt: usize,
    neg_log10_p_mut: usize,
}

impl MethBin {
    pub fn from(s: &MethylatedSite, strand: &Strand) -> Self {
        let pmin = 1e-4;
        let (wt_unmeth, wt_meth, mut_unmeth, mut_meth) = match strand {
            Strand::Forward => (
                s.wt_freq.get(Some(&Dna::C)) as f32,
                s.wt_freq.get(Some(&Dna::T)) as f32,
                s.mut_freq.get(Some(&Dna::C)) as f32,
                s.mut_freq.get(Some(&Dna::T)) as f32,
            ),
            Strand::Backward => (
                s.wt_freq.get(Some(&Dna::G)) as f32,
                s.wt_freq.get(Some(&Dna::A)) as f32,
                s.mut_freq.get(Some(&Dna::G)) as f32,
                s.mut_freq.get(Some(&Dna::A)) as f32,
            ),
        };

        Self {
            neg_log10_p_wt: Self::neg_log10_beta(wt_meth, wt_unmeth, pmin),
            neg_log10_p_mut: Self::neg_log10_beta(mut_meth, mut_unmeth, pmin),
        }
    }

    fn neg_log10_beta(meth: f32, unmeth: f32, pmin: f32) -> usize {
        (-(meth / (unmeth + meth).max(1.0)).max(pmin).log10()) as usize
    }
}

impl From<MethBin> for Box<str> {
    fn from(meth_bin: MethBin) -> Self {
        format!("{}\t{}", meth_bin.neg_log10_p_wt, meth_bin.neg_log10_p_mut).into_boxed_str()
    }
}

impl std::fmt::Display for MethBin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\t{}", self.neg_log10_p_wt, self.neg_log10_p_mut)
    }
}

/// Gene feature count statistics for different gene regions
pub struct GeneFeatureCount {
    five_prime: HashMap<MethBin, Vec<usize>>,
    cds: HashMap<MethBin, Vec<usize>>,
    three_prime: HashMap<MethBin, Vec<usize>>,
    non_coding: HashMap<MethBin, Vec<usize>>,
}

impl GeneFeatureCount {
    fn get_meth_bins(&self) -> Vec<MethBin> {
        let mut meth_bins: Vec<MethBin> = self
            .cds
            .iter()
            .map(|x| x.key().clone())
            .chain(self.five_prime.iter().map(|x| x.key().clone()))
            .chain(self.three_prime.iter().map(|x| x.key().clone()))
            .chain(self.non_coding.iter().map(|x| x.key().clone()))
            .collect();

        meth_bins.sort();
        meth_bins.dedup();
        meth_bins
    }

    fn max_count(map: &HashMap<MethBin, Vec<usize>>, k: &MethBin) -> usize {
        map.get(k)
            .and_then(|v| v.iter().cloned().max())
            .unwrap_or(0)
    }

    pub fn print(&self, max_width: usize) {
        fn print_row(label: &str, data: &[usize], scale: usize, max_width: usize) {
            for &n in data {
                let n1 = n.div_ceil(scale);
                let n0 = max_width.saturating_sub(n1);
                eprintln!(
                    "{:<6}{}{} {}",
                    label,
                    vec!["*"; n1].join(""),
                    vec![" "; n0].join(""),
                    n
                );
            }
        }

        let meth_bins = self.get_meth_bins();

        for k in &meth_bins {
            eprintln!("{}", k);

            let nmax = [
                GeneFeatureCount::max_count(&self.cds, &k),
                GeneFeatureCount::max_count(&self.five_prime, &k),
                GeneFeatureCount::max_count(&self.three_prime, &k),
                GeneFeatureCount::max_count(&self.non_coding, &k),
            ]
            .into_iter()
            .max()
            .unwrap();

            let scale = nmax.div_ceil(max_width);

            if let Some(x) = self.five_prime.get(k) {
                print_row("5'UTR", x.value(), scale, max_width);
            }
            if let Some(x) = self.cds.get(k) {
                print_row("CDS", x.value(), scale, max_width);
            }
            if let Some(x) = self.three_prime.get(k) {
                print_row("3'UTR", x.value(), scale, max_width);
            }
            if let Some(x) = self.non_coding.get(k) {
                print_row("ncRNA", x.value(), scale, max_width);
            }
        }
    }

    pub fn to_tsv(&self, file_path: &str) -> anyhow::Result<()> {
        fn into_boxed_str(label: &str, meth_bin: &MethBin, data: &[usize]) -> Vec<Box<str>> {
            data.iter()
                .enumerate()
                .map(|(i, &n)| format!("{}\t{}\t{}\t{}", label, i, meth_bin, n).into_boxed_str())
                .collect()
        }

        let mut writer = common_io::open_buf_writer(file_path)?;

        writer.write_all(
            b"#feature\tgenomic_bin\t-log10MAF(methylated)\t-log10MAF(background)\tcount\n",
        )?;

        let meth_bins = self.get_meth_bins();

        for k in &meth_bins {
            if let Some(data) = self.five_prime.get(k) {
                for l in into_boxed_str("5UTR", k, data.value()) {
                    writer.write_all(l.as_bytes())?;
                    writer.write_all(b"\n")?;
                }
            }
            if let Some(data) = self.three_prime.get(k) {
                for l in into_boxed_str("3UTR", k, data.value()) {
                    writer.write_all(l.as_bytes())?;
                    writer.write_all(b"\n")?;
                }
            }
            if let Some(data) = self.cds.get(k) {
                for l in into_boxed_str("CDS", k, data.value()) {
                    writer.write_all(l.as_bytes())?;
                    writer.write_all(b"\n")?;
                }
            }
            if let Some(data) = self.non_coding.get(k) {
                for l in into_boxed_str("ncRNA", k, data.value()) {
                    writer.write_all(l.as_bytes())?;
                    writer.write_all(b"\n")?;
                }
            }
        }

        writer.flush()?;
        Ok(())
    }
}

/// Trait for computing histogram statistics on methylation sites
pub trait Histogram {
    fn count_gene_features(
        &self,
        gff_file: &str,
        n_genomic_bins: usize,
    ) -> anyhow::Result<GeneFeatureCount>;

    fn count_on_feature_map(
        &self,
        feature_map: &HashMap<GeneId, GffRecord>,
        n_genomic_bins: usize,
    ) -> HashMap<MethBin, Vec<usize>>;
}

impl Histogram for HashMap<GeneId, Vec<MethylatedSite>> {
    fn count_gene_features(
        &self,
        gff_file: &str,
        n_genomic_bins: usize,
    ) -> anyhow::Result<GeneFeatureCount> {
        let gff_records = read_gff_record_vec(gff_file)?;

        // Separate protein-coding and non-coding records
        let protein_coding_records: Vec<GffRecord> = gff_records
            .iter()
            .filter(|rec| rec.gene_type == GeneType::CodingGene)
            .cloned()
            .collect();

        let non_coding_records: Vec<GffRecord> = gff_records
            .iter()
            .filter(|rec| rec.gene_type != GeneType::CodingGene)
            .cloned()
            .collect();

        let UnionGeneModel {
            gene_boundaries: _,
            cds,
            five_prime_utr,
            three_prime_utr,
        } = build_union_gene_model(&protein_coding_records)?;

        let UnionGeneModel {
            gene_boundaries: nc_gene_boundaries,
            cds: _,
            five_prime_utr: _,
            three_prime_utr: _,
        } = build_union_gene_model(&non_coding_records)?;

        let n_five_prime = five_prime_utr.take_max_length().max(10);
        let n_cds = cds.take_max_length();
        let n_three_prime = three_prime_utr.take_max_length().max(20);
        let ntot = n_five_prime + n_cds + n_three_prime;

        let nbins_five_prime = n_five_prime as usize * n_genomic_bins / ntot as usize;
        let nbins_cds = n_cds as usize * n_genomic_bins / ntot as usize;
        let nbins_three_prime = n_three_prime as usize * n_genomic_bins / ntot as usize;

        let five_prime = self.count_on_feature_map(&five_prime_utr, nbins_five_prime);
        let cds = self.count_on_feature_map(&cds, nbins_cds);
        let three_prime = self.count_on_feature_map(&three_prime_utr, nbins_three_prime);
        let non_coding = self.count_on_feature_map(&nc_gene_boundaries, n_genomic_bins);

        Ok(GeneFeatureCount {
            five_prime,
            cds,
            three_prime,
            non_coding,
        })
    }

    fn count_on_feature_map(
        &self,
        gene_gff_map: &HashMap<GeneId, GffRecord>,
        n_genomic_bins: usize,
    ) -> HashMap<MethBin, Vec<usize>> {
        let ret = HashMap::new();

        self.iter().for_each(|x| {
            let g = x.key();
            let sites = x.value();

            if let Some(gff) = gene_gff_map.get(&g) {
                // Convert GFF coordinates (1-based, inclusive) to 0-based half-open [lb, ub)
                // to match BAM/m6a_pos coordinates
                let lb = (gff.start - 1).max(0); // GFF 1-based start -> 0-based
                let ub = gff.stop; // GFF 1-based inclusive end -> 0-based exclusive end
                let length = (ub - lb).max(1) as usize;

                for s in sites.iter() {
                    if s.m6a_pos < ub && s.m6a_pos >= lb {
                        let beta_bin = MethBin::from(s, &gff.strand);

                        let mut entry = ret.entry(beta_bin).or_insert(vec![0; n_genomic_bins]);
                        let genomic = entry.value_mut();

                        // relative position with respect to (lb and ub)
                        let rel_pos = (match gff.strand {
                            Strand::Forward => (s.m6a_pos - lb) as usize,
                            Strand::Backward => (ub - s.m6a_pos - 1) as usize,
                        } * n_genomic_bins)
                            / length;

                        genomic[rel_pos] += 1;
                    }
                }
            }
        });
        ret
    }
}
