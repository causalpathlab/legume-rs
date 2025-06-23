// use matrix_util::common_io::read_lines_of_words_delim;
// use noodles::gff as noodles_gff;

// pub struct GffVec<'a> {
//     data: Vec<noodles_gff::Record<'a>>,
// }

// impl<'a> GffVec<'a> {
//     pub fn from_file(file_path: &str) -> anyhow::Result<Self> {
//         let (lines_of_words, _header) = read_lines_of_words_delim(file_path, vec!['\t', ','], -1)?;

//         unimplemented!("");

//         // Ok(Self {
//         //     data: lines_of_words
//         //         .into_iter()
//         //         .filter_map(|words| parse_gff(words))
//         //         .collect(),
//         // })
//     }
// }

// /// Parse a GFF line to a record
// ///
// /// https://en.wikipedia.org/wiki/General_feature_format
// ///
// fn parse_gff(words: Vec<Box<str>>) -> Option<noodles_gff::Record> {
//     const SEP_ATTR: char = ':';
//     const NUM_FIELDS: usize = 9;

//     if words.len() == NUM_FIELDS {
// 	unimplemented!("");
//         // let mut rec = noodles_gff::Record::new();
//         // *rec.seqname_mut() = words[0].to_string();
//         // *rec.source_mut() = words[1].to_string();
//         // *rec.feature_type_mut() = words[2].to_string();
//         // *rec.start_mut() = words[3].parse().unwrap_or(0);
//         // *rec.end_mut() = words[4].parse().unwrap_or(0);
//         // *rec.score_mut() = words[5].to_string();
//         // *rec.strand_mut() = words[6].to_string();
//         // *rec.phase_mut() = match words[7].as_ref() {
//         //     "." => gff::Phase::default(),
//         //     _ => gff::Phase::from(words[7].parse().unwrap_or(0u8)),
//         // };

//         // for z in words[8].split_whitespace() {
//         //     let kv: Vec<&str> = z.split(SEP_ATTR).collect();
//         //     if kv.len() == 2 {
//         //         rec.attributes_mut()
//         //             .insert(kv[0].to_string(), kv[1].to_string());
//         //     }
//         // }
//         // Some(rec)
//     } else {
//         None
//     }
// }

// // impl std::ops::Index<usize> for GffVec {
// //     type Output = gff::Record;
// //     fn index(&self, idx: usize) -> &Self::Output {
// //         &self.data[idx]
// //     }
// // }

// // impl IntoIterator for GffVec {
// //     type Item = gff::Record;
// //     type IntoIter = std::vec::IntoIter<Self::Item>;

// //     fn into_iter(self) -> Self::IntoIter {
// //         self.data.into_iter()
// //     }
// // }

// // impl<'a> IntoIterator for &'a GffVec {
// //     type Item = &'a gff::Record;
// //     type IntoIter = std::slice::Iter<'a, gff::Record>;

// //     fn into_iter(self) -> Self::IntoIter {
// //         self.data.iter()
// //     }
// // }
