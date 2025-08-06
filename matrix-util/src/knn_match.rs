use std::collections::HashMap;
use std::fmt::{Debug, Display};

/// A dictionary (HnswMap wrapper) for fast column look-up
///
pub struct ColumnDict<K> {
    pub dict: instant_distance::HnswMap<VecPoint, K>,
    pub data_vec: Vec<VecPoint>,
    pub name2index: HashMap<K, usize>,
}

impl<K> ColumnDict<K>
where
    K: Clone + Eq + std::hash::Hash + Debug + Display + std::cmp::PartialEq,
{
    pub fn names(&self) -> &Vec<K> {
        &self.dict.values
    }

    pub fn from_ndarray_views<'a>(data: Vec<ndarray::ArrayView1<'a, f32>>, names: Vec<K>) -> Self {
        <ColumnDict<K> as ColumnDictOps<K, ndarray::ArrayView1<'a, f32>>>::from_column_views(
            data, names,
        )
    }

    pub fn from_ndarray(data: ndarray::Array2<f32>, names: Vec<K>) -> Self {
        let views: Vec<_> = data.outer_iter().collect();
        Self::from_ndarray_views(views, names)
    }

    pub fn from_dvector_views(data: Vec<nalgebra::DVectorView<f32>>, names: Vec<K>) -> Self {
        <ColumnDict<K> as ColumnDictOps<K, nalgebra::DVectorView<f32>>>::from_column_views(
            data, names,
        )
    }

    pub fn from_dmatrix(data: nalgebra::DMatrix<f32>, names: Vec<K>) -> Self {
        Self::from_dvector_views(data.column_iter().collect(), names)
    }

    pub fn empty_ndarray_views() -> Self {
        <ColumnDict<K> as ColumnDictOps<K, ndarray::ArrayView1<f32>>>::empty()
    }

    pub fn empty_dvector_views() -> Self {
        <ColumnDict<K> as ColumnDictOps<K, nalgebra::DVectorView<f32>>>::empty()
    }

    pub fn dim(&self) -> Option<usize> {
        if self.data_vec.len() > 0 {
            Some(self.data_vec[0].len())
        } else {
            None
        }
    }

    /// k-nearest neighbour match by name within the same dictionary
    /// to return a Vec of names
    ///
    /// * `query_name` - the name of the column to match
    /// * `knn` - the number of nearest neighbours to return
    ///
    pub fn search_others(&self, query_name: &K, knn: usize) -> anyhow::Result<(Vec<K>, Vec<f32>)> {
        self.search_by_query_name(query_name, knn, true)
    }

    /// k-nearest neighbour match by name within the same dictionary
    /// to return a Vec of names
    ///
    /// * `query_name` - the name of the column to match
    /// * `knn` - the number of nearest neighbours to return
    /// * `exclude_same` - exclude the same query name
    ///
    pub fn search_by_query_name(
        &self,
        query_name: &K,
        knn: usize,
        exclude_same: bool,
    ) -> anyhow::Result<(Vec<K>, Vec<f32>)> {
        use instant_distance::Search;

        let nquery = knn.min(self.data_vec.len());

        if let Some(self_idx) = self.name2index.get(query_name) {
            let query = &self.data_vec[*self_idx];

            let mut search = Search::default();
            let knn_iter = self.dict.search(query, &mut search).take(nquery);
            let mut points = Vec::with_capacity(nquery);
            let mut distances = Vec::with_capacity(nquery);
            for v in knn_iter {
                let vv = v.value;
                if exclude_same && vv == query_name {
                    continue;
                }
                let dd = v.distance;
                points.push(vv.clone());
                distances.push(dd);
            }
            Ok((points, distances))
        } else {
            Err(anyhow::anyhow!("name {} not found", query_name))
        }
    }

    /// k-nearest neighbour match by name within the same dictionary
    /// to return a Vec of names
    ///
    /// * `query` - query data `VecPoint`
    /// * `knn` - the number of nearest neighbours to return
    ///
    pub fn search_by_query_data(
        &self,
        query: &VecPoint,
        knn: usize,
    ) -> anyhow::Result<(Vec<K>, Vec<f32>)> {
        use instant_distance::Search;

        if self.dim().unwrap_or(0) != query.len() {
            return Err(anyhow::anyhow!("query's dim does not match"));
        }

        let nquery = knn.min(self.data_vec.len());

        let mut search = Search::default();
        let knn_iter = self.dict.search(query, &mut search).take(nquery);
        let mut points = Vec::with_capacity(nquery);
        let mut distances = Vec::with_capacity(nquery);
        for v in knn_iter {
            let vv = v.value;
            let dd = v.distance;
            points.push(vv.clone());
            distances.push(dd);
        }

        Ok((points, distances))
    }

    /// k-nearest neighbour match by name against another dictionary
    /// to return a Vec of names in the other dictionary
    ///
    /// * `query_name` - the name of the column to match
    /// * `knn` - the number of nearest neighbours to return
    /// * `against` - the dictionary to match against
    ///
    pub fn match_by_query_name_against(
        &self,
        query_name: &K,
        knn: usize,
        against: &Self,
    ) -> anyhow::Result<(Vec<K>, Vec<f32>)> {
        use instant_distance::Search;

        let nquery = knn.min(against.data_vec.len());

        if let Some(self_idx) = self.name2index.get(query_name) {
            let query = &self.data_vec[*self_idx];
            let mut search = Search::default();
            let knn_iter = against.dict.search(query, &mut search).take(nquery);
            let mut points = Vec::with_capacity(nquery);
            let mut distances = Vec::with_capacity(nquery);
            for v in knn_iter {
                let vv = v.value;
                let dd = v.distance;
                points.push(vv.clone());
                distances.push(dd);
            }
            Ok((points, distances))
        } else {
            Err(anyhow::anyhow!("name {} not found", query_name))
        }
    }
}

pub trait ColumnDictOps<K, V> {
    fn empty() -> Self;
    fn from_column_views(data: Vec<V>, names: Vec<K>) -> Self;
}

impl<T, V> ColumnDictOps<T, V> for ColumnDict<T>
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

        let data_vec: Vec<VecPoint> = (0..nn).map(|j| data[j].to_vp()).collect();

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
/// a wrapper for `Vec<f32>`
pub struct VecPoint {
    pub data: Vec<f32>,
}

impl VecPoint {
    pub fn len(&self) -> usize {
        self.data.len()
    }
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
    /// Euclidean distance
    fn distance(&self, other: &Self) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }
}
