#![allow(dead_code)]

pub use crate::gff::GeneId as Gene;
use std::collections::HashMap;

#[derive(Hash, Eq, PartialEq, Clone, PartialOrd, Ord)]
pub struct SiteInGene {
    pub gene: Gene,
    pub site: i64,
}

impl std::ops::Add<i64> for SiteInGene {
    type Output = Self;

    fn add(self, rhs: i64) -> Self::Output {
        Self {
            gene: self.gene,
            site: self.site + rhs,
        }
    }
}

impl std::ops::Sub<i64> for &SiteInGene {
    type Output = SiteInGene;

    fn sub(self, rhs: i64) -> Self::Output {
        SiteInGene {
            gene: self.gene.clone(),
            site: self.site - rhs,
        }
    }
}

pub trait Stratify {
    /// group `SiteinGene` into `Gene -> Sites`
    fn stratify_by_gene(&self) -> HashMap<Gene, Vec<SiteInGene>>;
}

impl Stratify for Vec<&SiteInGene> {
    fn stratify_by_gene(&self) -> HashMap<Gene, Vec<SiteInGene>> {
        let mut ret: HashMap<Gene, Vec<SiteInGene>> = HashMap::new();
        for &sg in self.iter() {
            ret.entry(sg.gene.clone()).or_default().push(sg.clone());
        }
        for sites in ret.values_mut() {
            sites.sort();
            sites.dedup();
        }
        ret
    }
}
