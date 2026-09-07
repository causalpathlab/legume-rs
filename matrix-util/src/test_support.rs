//! Fixture helpers shared by the io test modules.

use std::io::Write;

/// Write `content` to a scratch file named `name` inside `dir`.
pub fn scratch(dir: &tempfile::TempDir, name: &str, content: &str) -> String {
    let path = dir.path().join(name);
    std::fs::write(&path, content).expect("write scratch file");
    path.to_string_lossy().into_owned()
}

/// Single-member gzip of `content`, what `gzip` itself writes.
pub fn scratch_gz(dir: &tempfile::TempDir, name: &str, content: &str) -> String {
    let path = dir.path().join(name);
    write_gz_members(&path, content, 1, false);
    path.to_string_lossy().into_owned()
}

/// Encode `text` as `members` concatenated gzip members, optionally giving each
/// one a BGZF-style `FEXTRA` field. Multi-member gzip is what `bgzip` and the
/// pipelines that post-process a file with it produce. With `members == 1` the
/// bytes of `text` go through unchanged.
pub fn write_gz_members(path: &std::path::Path, text: &str, members: usize, bgzf_extra: bool) {
    assert!(members > 0);
    let payloads: Vec<String> = if members == 1 {
        vec![text.to_string()]
    } else {
        let lines: Vec<&str> = text.lines().collect();
        let per = lines.len().div_ceil(members).max(1);
        lines
            .chunks(per)
            .map(|chunk| {
                let mut payload = chunk.join("\n");
                payload.push('\n');
                payload
            })
            .collect()
    };

    let mut bytes: Vec<u8> = Vec::new();
    for payload in payloads {
        let mut builder = flate2::GzBuilder::new();
        if bgzf_extra {
            // The `BC` subfield BGZF puts in every block: id, 2-byte length,
            // then the block size. Its value is never read back here -- what
            // matters is that the member carries an extra field at all.
            builder = builder.extra(vec![b'B', b'C', 2, 0, 0, 0]);
        }
        let mut enc = builder.write(Vec::new(), flate2::Compression::default());
        enc.write_all(payload.as_bytes()).expect("write gz member");
        bytes.extend(enc.finish().expect("finish gz member"));
    }
    std::fs::write(path, bytes).expect("write gz file");
}
