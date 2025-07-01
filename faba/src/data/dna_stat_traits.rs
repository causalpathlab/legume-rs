#![allow(dead_code)]

use crate::data::gff::*;
use crate::data::positions::*;
use crate::data::sam::*;
use crate::data::visitors_htslib::*;

use rust_htslib::bam::{self, record::Aux};

pub trait DnaStatMap: VisitWithBamOps {
    fn add_bam_record(&mut self, bam_record: bam::Record, lb: i64, ub: i64);

    /// update DNA count statistics using the information within this region
    /// * `bam_file_path` - file path
    /// * `gff_record` - `GffRecord` read from an annotation file
    fn update_bam_by_gene(
        &mut self,
        bam_file_path: &str,
        gff_record: &GffRecord,
        gene_barcode_tag: &str,
    ) -> anyhow::Result<()> {
        self.visit_bam_by_gene(
            bam_file_path,
            gff_record,
            gene_barcode_tag,
            &Self::update_visitor_by_gene,
        )
    }

    /// update DNA count statistics using the information within this region
    /// * `bam_file_path` - file path
    /// * `region` - (chromosome, start, stop), [start, stop), zero-based
    fn update_bam_by_region(
        &mut self,
        bam_file_path: &str,
        region: (&str, i64, i64),
    ) -> anyhow::Result<()> {
        self.visit_bam_by_region(bam_file_path, region, &Self::update_visitor_by_region)
    }

    fn update_visitor_by_region(&mut self, region: (&str, i64, i64), bam_record: bam::Record) {
        let (_chr, lb, ub) = region;
        self.add_bam_record(bam_record, lb, ub);
    }

    fn update_visitor_by_gene(
        &mut self,
        gff_record: &GffRecord,
        gene_barcode_tag: &str,
        bam_record: bam::Record,
    ) {
        let gene_id = &gff_record.gene_id;
        let lb = gff_record.start;
        let ub = gff_record.stop;

        let gene_id_found = match bam_record.aux(gene_barcode_tag.as_bytes()) {
            Ok(Aux::String(id)) => match parse_ensembl_id(id) {
                Some(id) => Gene::Ensembl(id.into()),
                _ => Gene::Missing,
            },
            _ => Gene::Missing,
        };

        if gene_id_found == *gene_id {
            self.add_bam_record(bam_record, lb, ub);
        }
    }

    /// output cell barcodes found
    fn cells(&self) -> Vec<&CellBarcode>;

    /// output sorted genomic positions
    fn sorted_positions(&self) -> Vec<i64>;
}
