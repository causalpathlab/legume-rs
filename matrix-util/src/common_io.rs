#![allow(dead_code)]

use flate2::read::GzDecoder;
use rayon::prelude::*;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use tempfile::tempdir;

/// Define a Delimiter enum to handle both &str and `Vec<char>`
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

impl From<&[char]> for Delimiter {
    fn from(chars: &[char]) -> Self {
        Delimiter::Chars(chars.to_vec())
    }
}

impl<const N: usize> From<&[char; N]> for Delimiter {
    fn from(chars: &[char; N]) -> Self {
        Delimiter::Chars(chars.to_vec())
    }
}

///
/// Read every line of the input_file into memory
///
/// * `input_file` - file name--either gzipped or not
///
pub fn read_lines(input_file_path: &str) -> anyhow::Result<Vec<Box<str>>> {
    let buf: Box<dyn BufRead> = open_buf_reader(input_file_path)?;
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
pub fn write_lines(lines: &Vec<Box<str>>, output_file_path: &str) -> anyhow::Result<()> {
    write_types(lines, output_file_path)
}

///
/// Write every line into the output_file
///
/// * `lines` - vector of lines
/// * `output_file` - file name--either gzipped or not
///
pub fn write_types<T>(lines: &Vec<T>, output_file_path: &str) -> anyhow::Result<()>
where
    T: std::fmt::Display,
{
    let mut buf = open_buf_writer(output_file_path)?;
    for line in lines {
        if let Err(e) = writeln!(buf, "{}", line) {
            if e.kind() == std::io::ErrorKind::BrokenPipe {
                return Ok(());
            } else {
                return Err(anyhow::anyhow!("unexpected error: {}", e));
            }
        }
    }
    buf.flush()?;
    Ok(())
}

pub struct ReadLinesOut<T: Send> {
    pub lines: Vec<Vec<T>>,
    pub header: Vec<Box<str>>,
}

///
/// Generic function to read lines and parse them into a vector of words or types.
///
/// * `input_file` - file name--either gzipped or not
/// * `hdr_line` - location of a header line (-1 = no header line)
/// * `parse_fn` - function to parse each line into the desired type
///
pub fn read_lines_of_words_generic<T>(
    input_file: &str,
    hdr_line: i64,
    parse_header_fn: impl Fn(&str) -> Vec<Box<str>> + Sync,
    parse_fn: impl Fn(&str) -> Vec<T> + Sync,
) -> anyhow::Result<ReadLinesOut<T>>
where
    T: Send,
{
    let buf_reader: Box<dyn BufRead> = open_buf_reader(input_file)?;

    fn is_not_comment_line(line: &str) -> bool {
        if line.starts_with('#') || line.starts_with('%') {
            return false;
        }
        true
    }

    let lines_raw: Vec<Box<str>> = buf_reader
        .lines()
        .map_while(Result::ok)
        .map(|x| x.into_boxed_str())
        .filter(|x| is_not_comment_line(x.as_ref()))
        .collect();

    let mut header = vec![];

    // Parsing takes more time, so split them into parallel jobs
    let mut lines: Vec<(usize, Vec<T>)> = if hdr_line < 0 {
        lines_raw
            .iter()
            .enumerate()
            .par_bridge()
            .map(|(i, s)| (i, parse_fn(s)))
            .collect()
    } else {
        let n_skip = hdr_line as usize;
        if lines_raw.len() < (n_skip + 1) {
            return Err(anyhow::anyhow!("not enough data"));
        }

        header.extend(parse_header_fn(&lines_raw[n_skip]));

        lines_raw[(n_skip + 1)..]
            .iter()
            .enumerate()
            .par_bridge()
            .map(|(i, s)| (i, parse_fn(s)))
            .collect()
    };

    if lines.len() > 100_000 {
        lines.par_sort_by_key(|&(i, _)| i);
    } else {
        lines.sort_by_key(|&(i, _)| i);
    }

    let lines = lines.into_iter().map(|(_, x)| x).collect();
    Ok(ReadLinesOut { lines, header })
}

///
/// Specialized function to read lines and parse them into a vector of types.
///
/// * `input_file` - file name--either gzipped or not
/// * `delim` - delimiter
/// * `hdr_line` - location of a header line (-1 = no header line)
///
pub fn read_lines_of_types<T>(
    input_file: &str,
    delim: impl Into<Delimiter>,
    hdr_line: i64,
) -> anyhow::Result<ReadLinesOut<T>>
where
    T: Send + std::str::FromStr + std::fmt::Display,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let delim = delim.into(); // Convert the input delimiter into the Delimiter enum

    let parse_fn = move |line: &str| -> Vec<T> {
        match &delim {
            Delimiter::Str(s) => line
                .split(s.as_str())
                .map(|x| x.parse::<T>().expect("failed to parse"))
                .collect(),
            Delimiter::Chars(chars) => line
                .split(chars.as_slice())
                .map(|x| x.parse::<T>().expect("failed to parse"))
                .collect(),
        }
    };

    let parse_header_fn = |line: &str| -> Vec<Box<str>> {
        line.split_whitespace()
            .map(|x| x.to_owned().into_boxed_str())
            .collect()
    };

    read_lines_of_words_generic(input_file, hdr_line, parse_header_fn, parse_fn)
}

///
/// Specialized function to read lines and parse them into a vector of words.
///
/// * `input_file` - file name--either gzipped or not
/// * `hdr_line` - location of a header line (-1 = no header line)
///
pub fn read_lines_of_words(
    input_file: &str,
    hdr_line: i64,
) -> anyhow::Result<ReadLinesOut<Box<str>>> {
    let parse_fn = |line: &str| -> Vec<Box<str>> {
        line.split_whitespace()
            .map(|x| x.to_owned().into_boxed_str())
            .collect()
    };

    read_lines_of_words_generic(input_file, hdr_line, parse_fn, parse_fn)
}

///
/// Specialized function to read lines and parse them into a vector of words.
///
/// * `input_file` - file name--either gzipped or not
/// * `delim` - delimiter
/// * `hdr_line` - location of a header line (-1 = no header line)
///
pub fn read_lines_of_words_delim(
    input_file: &str,
    delim: impl Into<Delimiter>,
    hdr_line: i64,
) -> anyhow::Result<ReadLinesOut<Box<str>>> {
    let delim = delim.into(); // Convert the input delimiter into the Delimiter enum

    let parse_fn = |line: &str| -> Vec<Box<str>> {
        match &delim {
            Delimiter::Str(s) => line
                .split(s.as_str())
                .map(|x| x.to_owned().into_boxed_str())
                .collect(),
            Delimiter::Chars(chars) => line
                .split(chars.as_slice())
                .map(|x| x.to_owned().into_boxed_str())
                .collect(),
        }
    };

    read_lines_of_words_generic(input_file, hdr_line, parse_fn, parse_fn)
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
    // we can simply override with stdout
    if output_file.eq_ignore_ascii_case("stdout") {
        return Ok(Box::new(std::io::BufWriter::new(std::io::stdout())));
    }

    if output_file.eq_ignore_ascii_case("stderr") {
        return Ok(Box::new(std::io::BufWriter::new(std::io::stderr())));
    }

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
    fn into_boxed_str(&self) -> Box<str>;
}

impl ToStr for Path {
    fn into_boxed_str(&self) -> Box<str> {
        self.to_str()
            .expect("failed to convert to string")
            .to_string()
            .into_boxed_str()
    }
}

impl ToStr for OsStr {
    fn into_boxed_str(&self) -> Box<str> {
        self.to_str()
            .expect("failed to convert to string")
            .to_string()
            .into_boxed_str()
    }
}

/// Unzip `zip_path` into the `extract_path` If `extract_path` is
/// `None`, just use a current directory.
/// * Returns `extract_path`
pub fn unzip_dir(zip_path: &str, extract_path: Option<&str>) -> anyhow::Result<Box<str>> {
    let zip_file = std::fs::File::open(&zip_path)?;
    let mut archive = zip::ZipArchive::new(zip_file)?;

    let extract_path = extract_path
        .map(|x| std::path::PathBuf::from(x))
        .unwrap_or(std::env::current_dir()?);

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let out_path = extract_path.join(file.name());
        // println!("{}", out_path.to_str().unwrap());
        if file.is_dir() {
            std::fs::create_dir_all(&out_path)?;
        } else {
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut outfile = std::fs::File::create(&out_path)?;
            std::io::copy(&mut file, &mut outfile)?;
        }
    }

    Ok(extract_path.into_boxed_str())
}

///
/// Take the parent directory, basename, and extension of a file
/// * `file` - file name
///
pub fn dir_base_ext(file_path: &str) -> anyhow::Result<(Box<str>, Box<str>, Box<str>)> {
    let path = Path::new(file_path);

    if let (Some(dir), Some(base), Some(ext)) = (path.parent(), path.file_stem(), path.extension())
    {
        Ok((
            dir.into_boxed_str(),
            base.into_boxed_str(),
            ext.into_boxed_str(),
        ))
    } else {
        Err(anyhow::anyhow!(
            "fail to parse dir, base, ext: {}",
            file_path
        ))
    }
}

///
/// Take the basename of a file
/// * `file` - file name
///
pub fn basename(file: &str) -> anyhow::Result<Box<str>> {
    let path = Path::new(file);
    if let Some(base) = path.file_stem() {
        Ok(base.into_boxed_str())
    } else {
        Err(anyhow::anyhow!("no file stem"))
    }
}

///
/// Take the extension of a file
/// * `file` - file name
///
pub fn extension(file: &str) -> anyhow::Result<Box<str>> {
    let path = Path::new(file);
    if let Some(ext) = path.extension() {
        Ok(ext.into_boxed_str())
    } else {
        Err(anyhow::anyhow!("failed to extract extension"))
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
        remove_file(file)?;
    }
    Ok(())
}
