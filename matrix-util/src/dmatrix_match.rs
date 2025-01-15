use crate::dmatrix_util::*;
use rayon::prelude::*;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// use crate::traits::*;
// use instant_distance as inst;

/// A dictionary (HnswMap wrapper) for fast column look-up
///
pub struct ColumnDict<T> {
    dict: instant_distance::HnswMap<VecPoint, T>,
    data_vec: Vec<VecPoint>,
    name2index: HashMap<T, usize>,
}

#[allow(dead_code)]
impl<T> ColumnDict<T>
where
    T: Clone + Eq + std::hash::Hash,
{
    pub fn empty() -> Self {
        use instant_distance::Builder;
        Self {
            dict: Builder::default().build(vec![], vec![]),
            data_vec: vec![],
            name2index: HashMap::new(),
        }
    }

    pub fn from_column_views(data: &Vec<nalgebra::DVectorView<f32>>, names: &Vec<T>) -> Self {
        let nn = data.len();

        debug_assert!(
            nn == names.len(),
            "Data and names must have the same length"
        );

        // debug_assert!(nn > 0, "Data must have at least one column");
        // let dd = data[0].len();

        let mut data_vec = vec![];
        let arc_data_vec = Arc::new(Mutex::new(&mut data_vec));

        (0..nn).into_iter().par_bridge().for_each(|j| {
            let v = data[j].iter().cloned().collect();
            arc_data_vec.lock().unwrap().push(VecPoint { data: v });
        });

        let mut name2index = HashMap::<T, usize>::new();

        for j in 0..nn {
            name2index.insert(names[j].clone(), j);
        }

        use instant_distance::Builder;
        let dict = Builder::default().build(data_vec.clone(), names.clone());

        let ret = ColumnDict {
            dict,
            data_vec,
            name2index,
        };

        ret
    }

    pub fn from_matrix(data: &DMatrix<f32>, names: &Vec<T>) -> Self {
        let nn = data.ncols();
        debug_assert!(
            nn == names.len(),
            "Data and names must have the same length"
        );

        let mut data_vec = vec![];
        let arc_data_vec = Arc::new(Mutex::new(&mut data_vec));

        (0..nn).into_iter().par_bridge().for_each(|j| {
            let mut v = Vec::with_capacity(data.nrows());
            for i in 0..data.nrows() {
                v.push(data[(i, j)]);
            }
            arc_data_vec.lock().unwrap().push(VecPoint { data: v });
        });

        let mut name2index = HashMap::<T, usize>::new();

        for j in 0..nn {
            name2index.insert(names[j].clone(), j);
        }

        use instant_distance::Builder;
        let dict = Builder::default().build(data_vec.clone(), names.clone());

        let ret = ColumnDict {
            dict,
            data_vec,
            name2index,
        };

        #[cfg(debug_assertions)]
        {
            // check if the order matches
            for (j, x) in ret.names().iter().enumerate() {
                if let Some(&i) = ret.name2index.get(x) {
                    debug_assert_eq!(i, j);
                }
            }
        }
        ret
    }

    pub fn names(&self) -> &Vec<T> {
        &self.dict.values
    }

    /// k-nearest neighbour match by name against another dictionary
    /// to return a Vec of names in the other dictionary
    ///
    /// * `name` - the name of the column to match
    /// * `knn` - the number of nearest neighbours to return
    /// * `against` - the dictionary to match against
    ///
    pub fn match_against_by_name<'a>(
        &'a self,
        name: &T,
        knn: usize,
        against: &'a Self,
    ) -> anyhow::Result<Vec<T>> {
        use instant_distance::Search;

        let nquery = knn.min(against.data_vec.len());

        if let Some(self_idx) = self.name2index.get(name) {
            let query = &self.data_vec[*self_idx];
            let mut search = Search::default();
            let knn_iter = against.dict.search(query, &mut search).take(nquery);
            let mut ret = vec![];
            for v in knn_iter {
                let vv = v.value;
                ret.push(vv.clone());
            }
            Ok(ret)
        } else {
            return Err(anyhow::anyhow!("Name not found"));
        }
    }
}

#[derive(Clone, Debug)]
struct VecPoint {
    data: Vec<f32>,
}

impl instant_distance::Point for VecPoint {
    // fn from
    fn distance(&self, other: &Self) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }
}
