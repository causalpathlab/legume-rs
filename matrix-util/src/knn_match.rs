use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::sync::{Arc, Mutex};

/// A dictionary (HnswMap wrapper) for fast column look-up
///
pub struct ColumnDict<T> {
    pub dict: instant_distance::HnswMap<VecPoint, T>,
    pub data_vec: Vec<VecPoint>,
    pub name2index: HashMap<T, usize>,
}

impl<T> ColumnDict<T>
where
    T: Clone + Eq + std::hash::Hash + Debug + Display,
{
    pub fn names(&self) -> &Vec<T> {
        &self.dict.values
    }

    pub fn from_ndarray_views<'a>(data: Vec<ndarray::ArrayView1<'a, f32>>, names: Vec<T>) -> Self {
        <ColumnDict<T> as ColumnDictOps<T, ndarray::ArrayView1<'a, f32>>>::from_column_views(
            data, names,
        )
    }

    pub fn from_dvector_views(data: Vec<nalgebra::DVectorView<f32>>, names: Vec<T>) -> Self {
        <ColumnDict<T> as ColumnDictOps<T, nalgebra::DVectorView<f32>>>::from_column_views(
            data, names,
        )
    }

    pub fn empty_ndarray_views<'a>() -> Self {
        <ColumnDict<T> as ColumnDictOps<T, ndarray::ArrayView1<'a, f32>>>::empty()
    }

    pub fn empty_dvector_views() -> Self {
        <ColumnDict<T> as ColumnDictOps<T, nalgebra::DVectorView<f32>>>::empty()
    }

    /// k-nearest neighbour match by name against another dictionary
    /// to return a Vec of names in the other dictionary
    ///
    /// * `query_name` - the name of the column to match
    /// * `knn` - the number of nearest neighbours to return
    /// * `against` - the dictionary to match against
    ///
    pub fn match_against_by_name(
        &self,
        query_name: &T,
        knn: usize,
        against: &Self,
    ) -> anyhow::Result<Vec<T>> {
        use instant_distance::Search;

        let nquery = knn.min(against.data_vec.len());

        if let Some(self_idx) = self.name2index.get(query_name) {
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
            return Err(anyhow::anyhow!("name {} not found", query_name));
        }
    }
}

pub trait ColumnDictOps<'a, T, V> {
    fn empty() -> Self;
    fn from_column_views(data: Vec<V>, names: Vec<T>) -> Self;
}

impl<'a, T, V> ColumnDictOps<'a, T, V> for ColumnDict<T>
where
    T: Clone + Eq + std::hash::Hash + Debug + Display,
    V: Sync + MakeVecPoint,
{
    fn empty() -> Self {
        use instant_distance::Builder;
        Self {
            dict: Builder::default().build(vec![], vec![]),
            data_vec: vec![],
            name2index: HashMap::new(),
        }
    }

    fn from_column_views(data: Vec<V>, names: Vec<T>) -> Self {
        let nn = data.len();

        debug_assert!(
            nn == names.len(),
            "Data and names must have the same length"
        );

        let mut data_vec = vec![];
        let arc_data_vec = Arc::new(Mutex::new(&mut data_vec));

        (0..nn)
            .into_par_iter()
            .progress_count(nn as u64)
            .for_each(|j| {
                arc_data_vec
                    .lock()
                    .expect("unable to lock")
                    .push(data[j].to_vp());
            });

        let mut name2index = HashMap::<T, usize>::new();

        names.iter().enumerate().for_each(|(j, x)| {
            name2index.insert(x.clone(), j);
        });

        use instant_distance::Builder;
        let dict = Builder::default().build(data_vec.clone(), names.clone());

        let ret = ColumnDict {
            dict,
            data_vec,
            name2index,
        };

        #[cfg(debug_assertions)]
        {
            // check if the name matches
            for x in ret.names().iter() {
                if let Some(&i) = ret.name2index.get(x) {
                    debug_assert_eq!(*x, names[i]);
                }
            }
        }
        ret
    }
}

#[derive(Clone, Debug)]
/// a wrapper for Vec<f32>
pub struct VecPoint {
    pub data: Vec<f32>,
}

pub trait MakeVecPoint {
    fn to_vp(&self) -> VecPoint;
}

impl MakeVecPoint for Vec<f32> {
    fn to_vp(&self) -> VecPoint {
        VecPoint { data: self.clone() }
    }
}

impl MakeVecPoint for nalgebra::DVectorView<'_, f32> {
    fn to_vp(&self) -> VecPoint {
        VecPoint {
            data: self.iter().cloned().collect(),
        }
    }
}

impl MakeVecPoint for ndarray::ArrayView1<'_, f32> {
    fn to_vp(&self) -> VecPoint {
        VecPoint {
            data: self.iter().cloned().collect(),
        }
    }
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
