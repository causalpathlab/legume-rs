use approx::assert_abs_diff_eq;
use matrix_util::knn_match::*;
use matrix_util::traits::MatOps;

#[test]
fn create_from_column_views() -> anyhow::Result<()> {
    // let dict = vec![
    // 		vec![1.0, 2.0, 3.0],
    // 		vec![4.0, 5.0, 6.0],
    // 		vec![7.0, 8.0, 9.0],
    // 	];

    let dict = ColumnDict::<usize>::empty_dvector_views();
    let other = ColumnDict::<usize>::empty_dvector_views();



    // let neighbours = dict.match_against_by_name(&0, 10, &other)?;

    Ok(())
}
