use faba::data::gff::*;

#[test]
fn gff_io_test() -> anyhow::Result<()> {
    let gff_file = "apoe.gff.gz";
    let gff_records = read_gff_rocords(gff_file)?;

    let UTRMap {
        five_prime,
        three_prime,
    } = build_utr_map(&gff_records)?;

    println!("{:?}", five_prime);
    println!("{:?}", three_prime);

    Ok(())
}
