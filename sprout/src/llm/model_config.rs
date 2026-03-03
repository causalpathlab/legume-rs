use std::path::PathBuf;

/// Configuration for loading a model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub context_length: usize,
    pub use_flash_attention: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            tokenizer_path: PathBuf::new(),
            context_length: 4096,
            use_flash_attention: false,
        }
    }
}
