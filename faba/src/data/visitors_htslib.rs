#![allow(dead_code)]

use crate::data::gff::*;

use rust_htslib::bam::{self, Read};

pub trait VisitWithBamOps {
    fn visit_bam_by_region<Visitor>(
        &mut self,
        bam_file_path: &str,
        region: (&str, i64, i64),
        visitor: &Visitor,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(&mut Self, (&str, i64, i64), bam::Record),
    {
        let index_file = bam_file_path.to_string() + ".bai";
        let mut bam_reader = bam::IndexedReader::from_path_and_index(bam_file_path, &index_file)?;

        bam_reader.fetch(region)?;

        let bam_records: Vec<bam::Record> = bam_reader
            .records()
            .into_iter()
            .filter_map(Result::ok)
            .filter(|rec| !rec.is_duplicate())
            .collect();

        for rec in bam_records {
            visitor(self, region, rec);
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
            .into_iter()
            .filter_map(Result::ok)
            .filter(|rec| !rec.is_duplicate())
            .collect();

        for rec in bam_records {
            visitor(self, gff_record, gene_barcode_tag, rec);
        }

        Ok(())
    }
}

// pub fn visit_bam_by_region<Visitor, Shared>(
//     shared: &mut Shared,
//     bam_file_path: &str,
//     region: (&str, i64, i64),
//     visitor: &Visitor,
// ) -> anyhow::Result<()>
// where
//     Visitor: Fn(&mut Shared, (&str, i64, i64), bam::Record),
// {
//     let index_file = bam_file_path.to_string() + ".bai";
//     let mut bam_reader = bam::IndexedReader::from_path_and_index(bam_file_path, &index_file)?;

//     bam_reader.fetch(region)?;

//     let bam_records: Vec<bam::Record> = bam_reader
//         .records()
//         .into_iter()
//         .filter_map(Result::ok)
//         .filter(|rec| !rec.is_duplicate())
//         .collect();

//     for rec in bam_records {
//         visitor(shared, region, rec);
//     }

//     Ok(())
// }

// pub fn visit_bam_by_gene<Visitor, Shared>(
//     shared: &mut Shared,
//     bam_file_path: &str,
//     gff_record: &GffRecord,
//     target_barcode_tag: &str,
//     visitor: &Visitor,
// ) -> anyhow::Result<()>
// where
//     Visitor: Fn(&mut Shared, &GffRecord, &str, bam::Record),
// {
//     let region = (
//         gff_record.seqname.as_ref(),
//         gff_record.start,
//         gff_record.stop,
//     );

//     let index_file = bam_file_path.to_string() + ".bai";
//     let mut bam_reader = bam::IndexedReader::from_path_and_index(bam_file_path, &index_file)?;

//     bam_reader.fetch(region)?;

//     let bam_records: Vec<bam::Record> = bam_reader
//         .records()
//         .into_iter()
//         .filter_map(Result::ok)
//         .filter(|rec| !rec.is_duplicate())
//         .collect();

//     for rec in bam_records {
//         visitor(shared, gff_record, target_barcode_tag, rec);
//     }

//     Ok(())
// }
