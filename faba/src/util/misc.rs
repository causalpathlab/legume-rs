#![allow(dead_code)]

/// paste words in a vector of `Box<str>` into `Box<str>`
///
/// * `words`
/// * `indices`
/// * `sep`
pub fn paste(words: &Vec<Box<str>>, indices: &Vec<usize>, sep: &str) -> Box<str> {
    let mut ret = String::new();
    let n = indices.len();
    for (i, &j) in indices.iter().enumerate() {
        if let Some(w) = words.get(j) {
            ret.push_str(w);
        }
        if n > 1 && i < (n - 1) {
            ret.push_str(sep);
        }
    }
    ret.into_boxed_str()
}
