#![allow(dead_code)]

use flate2::read::GzDecoder;
use rayon::prelude::*;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use tempfile::tempdir;

/// Define a Delimiter enum to handle both &str and Vec<char>
pub enum Delimiter {
    Str(String),
    Chars(Vec<char>),
}

impl From<&str> for Delimiter {
    fn from(s: &str) -> Self {
        Delimiter::Str(s.to_string())
    }
}

impl From<Vec<char>> for Delimiter {
    fn from(chars: Vec<char>) -> Self {
        Delimiter::Chars(chars)
    }
}

///
/// Read every line of the input_file into memory
///
/// * `input_file` - file name--either gzipped or not
///
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
pub fn write_lines(lines: &Vec<Box<str>>, output_file: &str) -> anyhow::Result<()> {
    let mut buf: Box<dyn Write> = open_buf_writer(output_file)?;
    for l in lines {
        writeln!(buf, "{}", l)?;
    }
    buf.flush()?;
    Ok(())
}

///
/// Write every line into the output_file
///
/// * `lines` - vector of lines
/// * `output_file` - file name--either gzipped or not
///
pub fn write_types<T>(lines: &Vec<T>, output_file: &str) -> anyhow::Result<()>
where
    T: std::str::FromStr + std::fmt::Display,
{
    let mut buf: Box<dyn Write> = open_buf_writer(output_file)?;
    for x in lines.iter() {
        writeln!(buf, "{}", x.to_string())?;
    }
    buf.flush()?;
    Ok(())
}

///
/// Read in each line by line, then parse each line into a vector or
/// words.
///
/// * `input_file` - file name--either gzipped or not
/// * `delim` - delimiter
/// * `hdr_line` - location of a header line (-1 = no header line)
///
pub fn read_lines_of_types<T>(
    input_file: &str,
    delim: impl Into<Delimiter>,
    hdr_line: i64,
) -> anyhow::Result<(Vec<Vec<T>>, Vec<Box<str>>)>
where
    T: Send + std::str::FromStr + std::fmt::Display,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let buf_reader: Box<dyn BufRead> = open_buf_reader(input_file)?;

    fn parse<T>(i: usize, line: &String, delim: &Delimiter) -> (usize, Vec<T>)
    where
        T: std::str::FromStr,
        <T as std::str::FromStr>::Err: std::fmt::Debug,
    {
        let parts = match delim {
            Delimiter::Str(s) => line.split(s).collect::<Vec<&str>>(),
            Delimiter::Chars(chars) => line.split(chars.as_slice()).collect::<Vec<&str>>(),
        };

        (
            i,
            parts
                .into_iter()
                .map(|x| x.parse::<T>().expect("failed to parse"))
                .collect(),
        )
    }

    fn is_not_comment_line(line: &String) -> bool {
        if line.starts_with('#') || line.starts_with('%') {
            return false;
        }
        true
    }

    let delim = delim.into(); // Convert the input delimiter into the Delimiter enum

    let lines_raw: Vec<String> = buf_reader
        .lines()
        .filter_map(|r| r.ok())
        .filter(is_not_comment_line)
        .collect();

    let mut hdr: Vec<Box<str>> = vec![];

    // Parsing takes more time, so split them into parallel jobs
    // Note: this will not make things sorted
    let mut lines: Vec<(usize, Vec<T>)> = if hdr_line < 0 {
        lines_raw
            .iter()
            .enumerate()
            .par_bridge()
            .map(|(i, s)| parse(i, s, &delim))
            .collect()
    } else {
        let n_skip = hdr_line as usize;
        if lines_raw.len() < (n_skip + 1) {
            return Err(anyhow::anyhow!("not enough data"));
        }

        hdr.extend(match &delim {
            Delimiter::Str(s) => lines_raw[n_skip]
                .split(s.as_str())
                .map(|x| x.to_owned().into_boxed_str())
                .collect::<Vec<_>>(),
            Delimiter::Chars(chars) => lines_raw[n_skip]
                .split(chars.as_slice())
                .map(|x| x.to_owned().into_boxed_str())
                .collect::<Vec<_>>(),
        });

        lines_raw[(n_skip + 1)..]
            .iter()
            .enumerate()
            .par_bridge()
            .map(|(i, s)| parse(i, s, &delim))
            .collect()
    };

    lines.sort_by_key(|&(i, _)| i);
    let lines = lines.into_iter().map(|(_, x)| x).collect();
    Ok((lines, hdr))
}

///
/// Read in each line by line, then parse each line into a vector or
/// words.
///
/// * `input_file` - file name--either gzipped or not
/// * `hdr_line` - location of a header line (-1 = no header line)
///
pub fn read_lines_of_words(
    input_file: &str,
    hdr_line: i64,
) -> anyhow::Result<(Vec<Vec<Box<str>>>, Vec<Box<str>>)> {
    // buffered reader
    let buf_reader: Box<dyn BufRead> = open_buf_reader(input_file)?;

    fn parse(i: usize, line: &String) -> (usize, Vec<Box<str>>) {
        let words: Vec<Box<str>> = line
            .split_whitespace()
            .map(|x| x.to_owned().into_boxed_str())
            .collect();
        (i, words)
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

    let mut hdr = vec![];

    // parsing takes more time, so split them into parallel jobs
    // note: this will not make things sorted
    let mut lines: Vec<(usize, Vec<Box<str>>)> = if hdr_line < 0 {
        lines_raw
            .iter()
            .enumerate()
            .par_bridge()
            .map(|(i, s)| parse(i, s))
            .collect()
    } else {
        let n_skip = hdr_line as usize;
        if lines_raw.len() < (n_skip + 1) {
            return Err(anyhow::anyhow!("not enough data"));
        }
        hdr.extend(parse(0, &lines_raw[n_skip]).1);
        lines_raw[(n_skip + 1)..]
            .iter()
            .enumerate()
            .par_bridge()
            .map(|(i, s)| parse(i, s))
            .collect()
    };

    lines.sort_by_key(|&(i, _)| i);
    let lines = lines.into_iter().map(|(_, x)| x).collect();
    Ok((lines, hdr))
}

///
/// Open a file for reading, and return a buffered reader
/// * `input_file` - file name--either gzipped or not
pub fn open_buf_reader(input_file: &str) -> anyhow::Result<Box<dyn BufRead>> {
    // take a look at the extension
    // return buffered reader accordingly
    let ext = Path::new(input_file).extension().and_then(|x| x.to_str());
    match ext {
        Some("gz") => {
            // dbg!(input_file);
            let input_file = File::open(input_file)?;
            let decoder = GzDecoder::new(input_file);
            Ok(Box::new(BufReader::new(decoder)))
        }
        _ => {
            // dbg!(input_file);
            let input_file = File::open(input_file)?;
            Ok(Box::new(BufReader::new(input_file)))
        }
    }
}

///
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

///
/// Create a directory if needed
/// * `file` - file name
///
pub fn mkdir(file: &str) -> anyhow::Result<()> {
    let path = Path::new(file);
    let dir = path.parent().ok_or(anyhow::anyhow!("no parent"))?;
    std::fs::create_dir_all(dir)?;
    Ok(())
}

trait ToStr {
    fn to_box_str(&self) -> Box<str>;
}

impl ToStr for Path {
    fn to_box_str(&self) -> Box<str> {
        self.to_str()
            .expect("failed to convert to string")
            .to_string()
            .into_boxed_str()
    }
}

impl ToStr for OsStr {
    fn to_box_str(&self) -> Box<str> {
        self.to_str()
            .expect("failed to convert to string")
            .to_string()
            .into_boxed_str()
    }
}

///
/// Take the basename of a file
/// * `file` - file name
///
pub fn dir_base_ext(file: &str) -> anyhow::Result<(Box<str>, Box<str>, Box<str>)> {
    let path = Path::new(file);

    if let (Some(dir), Some(base), Some(ext)) = (path.parent(), path.file_stem(), path.extension())
    {
        Ok((dir.to_box_str(), base.to_box_str(), ext.to_box_str()))
    } else {
        return Err(anyhow::anyhow!("no file stem"));
    }
}

///
/// Take the basename of a file
/// * `file` - file name
///
pub fn basename(file: &str) -> anyhow::Result<Box<str>> {
    let path = Path::new(file);
    if let Some(base) = path.file_stem() {
        Ok(base.to_box_str())
    } else {
        return Err(anyhow::anyhow!("no file stem"));
    }
}

///
/// Take the extension of a file
/// * `file` - file name
///
pub fn extension(file: &str) -> anyhow::Result<Box<str>> {
    let path = Path::new(file);
    if let Some(ext) = path.extension() {
        Ok(ext.to_box_str())
    } else {
        return Err(anyhow::anyhow!("failed to extract extension"));
    }
}

///
/// Create a temporary directory and suggest a file name
/// * `suffix` - suffix of the file name
///
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

///
/// Remove a file if it exists
/// * `file` - file name
///
pub fn remove_file(file: &str) -> anyhow::Result<()> {
    let path = Path::new(file);
    if path.exists() {
        if path.is_file() {
            std::fs::remove_file(path)?;
        } else {
            std::fs::remove_dir_all(path)?;
        }
    }
    Ok(())
}

///
/// Remove a file if it exists
/// * `files` - file name
///
pub fn remove_all_files(files: &Vec<Box<str>>) -> anyhow::Result<()> {
    for file in files {
        remove_file(&file)?;
    }
    Ok(())
}
