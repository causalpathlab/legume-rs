use flate2::read::GzDecoder;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use tempfile::tempdir;

///
/// Read every line of the input_file into memory
///
/// * `input_file` - file name--either gzipped or not
///
#[allow(dead_code)]
pub fn read_lines(input_file: &str) -> anyhow::Result<Vec<Box<str>>> {
    let buf: Box<dyn BufRead> = open_buf_reader(input_file)?;
    let mut lines = vec![];
    for x in buf.lines() {
        lines.push(x?.into_boxed_str());
    }
    Ok(lines)
}

///
/// Write every line into the output_file
///
/// * `lines` - vector of lines
/// * `output_file` - file name--either gzipped or not
///
#[allow(dead_code)]
pub fn write_lines(lines: &Vec<Box<str>>, output_file: &str) -> anyhow::Result<()> {
    let mut buf: Box<dyn Write> = open_buf_writer(output_file)?;
    for l in lines {
        writeln!(buf, "{}", l)?;
    }
    buf.flush()?;
    Ok(())
}

/// Read in each line by line, then parse each line into a vector or
/// words.
///
/// * `input_file` - file name--either gzipped or not
/// * `hdr_line` - location of a header line (-1 = no header line)
///
#[allow(dead_code)]
pub fn read_lines_of_words(
    input_file: &str,
    hdr_line: i64,
) -> anyhow::Result<(Vec<Vec<Box<str>>>, Vec<Box<str>>)> {
    // buffered reader
    let buf_reader: Box<dyn BufRead> = open_buf_reader(input_file)?;

    fn parse(line: &String) -> Vec<Box<str>> {
        let words: Vec<Box<str>> = line
            .split_whitespace()
            .map(|x| x.to_owned().into_boxed_str())
            .collect();
        words
    }

    fn is_not_comment_line(line: &String) -> bool {
        if line.starts_with('#') || line.starts_with('%') {
            return false;
        }
        true
    }

    let lines_raw: Vec<String> = buf_reader
        .lines()
        .filter_map(|r| r.ok())
        .filter(is_not_comment_line)
        .collect();

    let n_skip;
    let hdr;
    let lines;

    // parsing takes more time, so split them into parallel jobs
    // note: this will not make things sorted
    if hdr_line < 0 {
        hdr = Vec::<Box<str>>::new();
        lines = lines_raw.iter().par_bridge().map(parse).collect();
    } else {
        n_skip = hdr_line as usize;
        if lines_raw.len() < (n_skip + 1) {
            return Err(anyhow::anyhow!("not enough data"));
        }
        hdr = parse(&lines_raw[n_skip]);
        lines = lines_raw[(n_skip + 1)..]
            .iter()
            .par_bridge()
            .map(parse)
            .collect();
    }

    Ok((lines, hdr))
}

#[allow(dead_code)]
/// Open a file for reading, and return a buffered reader
/// * `input_file` - file name--either gzipped or not
pub fn open_buf_reader(input_file: &str) -> anyhow::Result<Box<dyn BufRead>> {
    // take a look at the extension
    // return buffered reader accordingly
    let ext = Path::new(input_file).extension().and_then(|x| x.to_str());
    match ext {
        Some("gz") => {
            dbg!(input_file);
            let input_file = File::open(input_file)?;
            let decoder = GzDecoder::new(input_file);
            Ok(Box::new(BufReader::new(decoder)))
        }
        _ => {
            dbg!(input_file);
            let input_file = File::open(input_file)?;
            Ok(Box::new(BufReader::new(input_file)))
        }
    }
}

#[allow(dead_code)]
/// Open a file for writing, and return a buffered writer
/// * `output_file` - file name--either gzipped or not
pub fn open_buf_writer(output_file: &str) -> anyhow::Result<Box<dyn std::io::Write>> {
    // take a look at the extension
    let ext = Path::new(output_file).extension().and_then(|x| x.to_str());
    match ext {
        Some("gz") => {
            let output_file = File::create(output_file)?;
            let encoder =
                flate2::write::GzEncoder::new(output_file, flate2::Compression::default());
            Ok(Box::new(BufWriter::new(encoder)))
        }
        _ => {
            let output_file = File::create(output_file)?;
            Ok(Box::new(BufWriter::new(output_file)))
        }
    }
}

#[allow(dead_code)]
/// Create a temporary directory and suggest a file name
/// * `suffix` - suffix of the file name
pub fn create_temp_dir_file(suffix: &str) -> anyhow::Result<std::path::PathBuf> {
    let temp_dir = tempdir()?.path().to_path_buf();
    std::fs::create_dir_all(&temp_dir)?;
    let temp_file = tempfile::Builder::new()
        .suffix(suffix)
        .tempfile_in(temp_dir)?
        .path()
        .to_owned();

    Ok(temp_file)
}
