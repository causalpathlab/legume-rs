use crate::common_io::write_lines;
use candle_core::Tensor;

#[allow(dead_code)]
pub fn write_tsv(file: &str, data: &Tensor) -> anyhow::Result<()> {
    let dims = data.dims();

    if dims.len() != 2 {
        return Err(anyhow::anyhow!("Expected 2 dimensions, got {}", dims.len()));
    }

    let lines: Vec<Box<str>> = (0..dims[0])
        .map(|i| {
            let row = data.narrow(0, i, 1).expect("failed to narrow in");
            let flatten_row = row.flatten_to(1).expect("flatten");
            let row_vec = flatten_row.to_vec1::<f32>().expect("to_vec1");
            row_vec
                .iter()
                .map(|&x| format!("{}", x))
                .collect::<Vec<_>>()
                .join("\t")
                .into_boxed_str()
        })
        .collect();

    write_lines(&lines, &file)?;

    Ok(())
}
