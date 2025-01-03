use crate::common_io::write_lines;
use ndarray::prelude::*;

#[allow(dead_code)]
pub fn write_tsv<T: std::fmt::Display>(file: &str, data: &Array2<T>) -> anyhow::Result<()>
where
    T: serde::Serialize,
{
    let lines: Vec<Box<str>> = data
        .rows()
        .into_iter()
        .map(|row| {
            let line = row
                .iter()
                .map(|x| format!("{}", x))
                .collect::<Vec<String>>()
                .join("\t");
            line.into_boxed_str()
        })
        .collect();

    write_lines(&lines, &file)?;
    Ok(())
}
