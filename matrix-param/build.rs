fn main() {
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    #[cfg(target_os = "linux")]
    {
        // No additional configuration needed for OpenBLAS
    }
}
