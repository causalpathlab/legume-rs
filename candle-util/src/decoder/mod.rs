//! Decoder modules for VAE-style topic / link-community models.
//!
//! - [`topic`]: dense multinomial / NB topic decoders
//! - [`embedded_topic`]: ETM-style embedded topic decoder
//! - [`delta_topic`]: delta-parameterized topic decoder
//! - [`joint_topic`]: paired/multi-view topic decoder
//! - [`nb_mixture`]: ambient-RNA NB mixture topic decoder
//! - [`poisson`]: simple Poisson decoder
//! - [`bipartite`]: bipartite link-community decoder
//! - [`dyn_decoder`]: trait-object wrapper for runtime decoder selection

pub mod bipartite;
pub mod delta_topic;
pub mod dyn_decoder;
pub mod embedded_topic;
pub mod joint_topic;
pub mod nb_mixture;
pub mod poisson;
pub mod topic;
