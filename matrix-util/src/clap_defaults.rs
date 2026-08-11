//! Clap-declared defaults for an `Args` struct.

/// `Default` for a clap `Args` struct, taken from **clap's own** declared
/// defaults rather than from `Default::default()` on each field.
///
/// This exists so a recorded fit configuration (`RunManifest::train_args`) can
/// gain a field without invalidating every model trained before it. The naive
/// `#[serde(default)]` fills a missing `epochs` with `0` and trains for zero
/// epochs; this fills it with `1000`, which is what the flag actually means.
/// Hand-written `Default` impls would work too, but they duplicate every
/// `default_value_t` and drift silently the moment one is changed — parsing an
/// empty argv back through the same derive cannot drift.
///
/// `required` arguments have no declared default, so they are relaxed and land
/// on an empty value. Every caller of this overwrites them (an update supplies
/// its own `--out` and inputs), and warm start independently re-checks the
/// architecture, so a wrong value here cannot reach training unnoticed.
pub fn clap_defaults<T: clap::Args + clap::FromArgMatches>() -> T {
    let cmd = T::augment_args(clap::Command::new("defaults")).mut_args(|a| {
        // Only `required` args need anything: everything else either declares a
        // default or is content to be absent. Handing a placeholder to, say, a
        // boolean flag would offer it a value it does not accept.
        let needs_placeholder = a.is_required_set()
            && a.get_action().takes_values()
            && a.get_default_values().is_empty();
        let a = a.required(false);
        if needs_placeholder {
            a.default_value("")
        } else {
            a
        }
    });
    let matches = cmd
        .try_get_matches_from(["defaults"])
        .expect("clap defaults: relaxing `required` should leave an always-parsable command");
    T::from_arg_matches(&matches)
        .expect("clap defaults: every argument has a value after relaxation")
}
