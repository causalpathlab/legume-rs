#![allow(dead_code)]

use genomic_data::bed::*;
use genomic_data::gff::*;

use rust_htslib::bam::{self, Read};

pub trait VisitWithBamOps {
    fn visit_bam_by_region<Visitor>(
        &mut self,
        bam_file_path: &str,
        bed: &Bed,
        visitor: &Visitor,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(&mut Self, &str, bam::Record),
    {
	let chr = bed.chr.as_ref();
        let region = (chr, bed.start, bed.stop);
        let index_file = bam_file_path.to_string() + ".bai";
        let mut bam_reader = bam::IndexedReader::from_path_and_index(bam_file_path, &index_file)?;

        bam_reader.fetch(region)?;

        let bam_records: Vec<bam::Record> = bam_reader
            .records()
            .filter_map(Result::ok)
            .filter(|rec| !rec.is_duplicate())
            .collect();

        for rec in bam_records {
            visitor(self, chr, rec);
        }

        Ok(())
    }

    fn visit_bam_by_gene<Visitor>(
        &mut self,
        bam_file_path: &str,
        gff_record: &GffRecord,
        gene_barcode_tag: &str,
        visitor: &Visitor,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(&mut Self, &GffRecord, &str, bam::Record),
    {
        let region = (
            gff_record.seqname.as_ref(),
            gff_record.start,
            gff_record.stop,
        );

        let index_file = bam_file_path.to_string() + ".bai";
        let mut bam_reader = bam::IndexedReader::from_path_and_index(bam_file_path, &index_file)?;

        bam_reader.fetch(region)?;

        let bam_records: Vec<bam::Record> = bam_reader
            .records()
            .filter_map(Result::ok)
            .filter(|rec| !rec.is_duplicate())
            .collect();

        for rec in bam_records {
            visitor(self, gff_record, gene_barcode_tag, rec);
        }

        Ok(())
    }
}
