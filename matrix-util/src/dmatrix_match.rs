use crate::dmatrix_util::*;
use rayon::prelude::*;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// use crate::traits::*;
// use instant_distance as inst;

/// A dictionary (HnswMap wrapper) for fast column look-up
///
struct ColumnDict<T> {
    dict: instant_distance::HnswMap<VecPoint, T>,
    data_vec: Vec<VecPoint>,
    name2index: HashMap<T, usize>,
}

#[allow(dead_code)]
impl<T> ColumnDict<T>
where
    T: Clone + Eq + std::hash::Hash,
{
    pub fn new(data: &DMatrix<f32>, names: &Vec<T>) -> Self {
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

        ColumnDict {
            dict,
            data_vec,
            name2index,
        }
    }

    pub fn names(&self) -> &Vec<T> {
        &self.dict.values
    }

    pub fn search_by_name<'a>(
        &'a self,
        name: &T,
        knn: usize,
        against: &'a mut Self,
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

// impl<T> ColumnMatch<T>
// where
//     T: nalgebra::RealField,
// {
//     fn new(data: DMatrix<T>) -> Self {
//         ColumnMatch { data }
//     }
// }
