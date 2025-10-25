// todo:

// use crate::common::*;
// use crate::data::dna::DnaBaseCount;
// use dashmap::DashMap as HashMap;
// use dashmap::DashSet as HashSet;

// fn collect_frequencies_nearby(
//     gene_sites: &HashMap<GeneId, Vec<MethylatedSite>>,
//     gff_map: &GffRecordMap,
//     bam_files: &[Box<str>],
// ) -> anyhow::Result<()> {
//     // Avoid double counting. Create intervals for each chromosome by
//     // scanning them all
//     let nsites = gene_sites.iter().map(|x| x.len()).sum::<usize>();
//     let chr_forward_sites: HashMap<Box<str>, HashSet<MethylatedSite>> =
//         HashMap::with_capacity(nsites);
//     let chr_backward_sites: HashMap<Box<str>, HashSet<MethylatedSite>> =
//         HashMap::with_capacity(nsites);

//     gene_sites.par_iter().for_each(|gs| {
//         if let Some(gff) = gff_map.get(gs.key()) {
//             let chr = &gff.seqname;
//             let strand = &gff.strand;
//             let mut entry = if strand == &Strand::Forward {
//                 chr_forward_sites.entry(chr.clone()).or_default()
//             } else {
//                 chr_backward_sites.entry(chr.clone()).or_default()
//             };
//             let list = entry.value_mut();
//             for s in gs.value() {
//                 list.insert(s.clone());
//             }
//         }
//     });

//     let window: i64 = 4;

//     let forward_freq_map: HashMap<usize, DnaBaseCount> =
//         HashMap::with_capacity(2 * window as usize + 1);

//     for chr_sites in chr_forward_sites.iter() {
//         let chr = chr_sites.key();
//         let njobs = chr_sites.value().len();
//         info!("gathering DNA frequencies from {} sites in {}", njobs, chr);
//         chr_sites
//             .value()
//             .par_iter()
//             .progress_count(njobs as u64)
//             .for_each(|s| {
//                 let mut freq_map = DnaBaseFreqMap::new();
//                 let pos = s.m6a_pos;
//                 let bed = Bed {
//                     chr: chr.clone(),
//                     start: (pos - window).max(0),
//                     stop: pos + window + 1,
//                 };
//                 for bam_file in bam_files.as_ref() {
//                     freq_map
//                         .update_bam_by_region(bam_file, &bed)
//                         .expect("failed to read the bam file");
//                 }
//                 let freq_map = freq_map.marginal_frequency_map().expect("marginal freq");
//                 for (j, p) in ((pos - window)..(pos + window + 1)).enumerate() {
//                     if let Some(count) = freq_map.get(&p) {
//                         let mut prev_count = forward_freq_map.entry(j).or_default();
//                         *prev_count += count;
//                     }
//                 }
//             });
//     }

//     for j in 0..(2 * window + 2) as usize {
//         if let Some(x) = forward_freq_map.get(&j) {
//             println!("{:?}", x.value());
//         }
//     }

//     Ok(())
// }
