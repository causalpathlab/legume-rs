// #![allow(dead_code)]

use crate::cocoa_common::*;


/// Generate 
///
/// ```text
/// Y(g,j) ~ Poisson{ sum_t delta(g,t,S(j)) * beta(g,t) * theta(j,t) }
/// ```
///
/// ```text
/// ln delta(g,t,s) ~ W(s) * tau(g,t) + sum_c X(s,c) * kappa(c) + eps
/// ```
///
/// ```text
/// logit W(s) ~ sum_c X(s,c) * omega(c) + eps
/// ```
/// 
pub fn generate_diff_data() {
// todo:
}
