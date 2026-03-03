use crate::llm::backend::{GenerateConfig, LlmBackend};
use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use std::path::Path;
use tokenizers::Tokenizer;

/// Candle-based LLM for local inference using GGUF models
pub struct CandleLlm {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    context_length: usize,
    model_name: String,
}

impl CandleLlm {
    /// Load a GGUF model from file
    pub fn from_gguf(
        model_path: &Path,
        tokenizer_path: &Path,
        cpu_only: bool,
        context_length: usize,
    ) -> Result<Self> {
        // Select device
        let device = if cpu_only {
            Device::Cpu
        } else {
            Self::select_device()?
        };

        log::info!("Using device: {:?}", device);

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Load GGUF model
        log::info!("Loading model from: {}", model_path.display());

        let mut file = std::fs::File::open(model_path)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)?;

        let model_name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        log::info!("Model loaded successfully: {}", model_name);

        Ok(Self {
            model,
            tokenizer,
            device,
            context_length,
            model_name,
        })
    }

    /// Select the best available device
    fn select_device() -> Result<Device> {
        // Try Metal on macOS
        #[cfg(target_os = "macos")]
        {
            if let Ok(device) = Device::new_metal(0) {
                log::info!("Using Metal GPU");
                return Ok(device);
            }
        }

        // Try CUDA on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                log::info!("Using CUDA GPU");
                return Ok(device);
            }
        }

        // Fall back to CPU
        log::info!("Using CPU");
        Ok(Device::Cpu)
    }

    /// Get the EOS token ID
    fn get_eos_token(&self) -> u32 {
        // Try various common EOS token names
        self.tokenizer
            .token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("<|end|>"))
            .or_else(|| self.tokenizer.token_to_id("<|eot_id|>"))
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(2)
    }
}

impl LlmBackend for CandleLlm {
    fn generate(&mut self, prompt: &str, config: &GenerateConfig) -> Result<String> {
        // Tokenize input
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;

        let prompt_tokens = encoding.get_ids().to_vec();
        let prompt_len = prompt_tokens.len();

        if prompt_len >= self.context_length {
            return Err(anyhow!(
                "Prompt too long: {} tokens (max: {})",
                prompt_len,
                self.context_length
            ));
        }

        // Create logits processor for sampling
        let mut logits_processor = LogitsProcessor::new(
            42, // seed
            Some(config.temperature as f64),
            Some(config.top_p as f64),
        );

        let eos_token_id = self.get_eos_token();
        let max_new_tokens = config.max_tokens.min(self.context_length - prompt_len);

        // Process the initial prompt
        let input_tensor = Tensor::new(prompt_tokens.as_slice(), &self.device)?
            .unsqueeze(0)?; // Shape: (1, prompt_len)

        let logits = self.model.forward(&input_tensor, 0)?;
        let logits = logits.squeeze(0)?; // Shape: (vocab_size,)
        let logits = logits.to_dtype(DType::F32)?;

        // Sample first token
        let mut next_token = logits_processor.sample(&logits)?;
        let mut generated_tokens = vec![next_token];

        // Check for immediate EOS
        if next_token == eos_token_id {
            return Ok(String::new());
        }

        // Generate remaining tokens
        let mut current_pos = prompt_len;

        for _ in 1..max_new_tokens {
            // Create input for next token
            let input_tensor = Tensor::new(&[next_token], &self.device)?
                .unsqueeze(0)?; // Shape: (1, 1)

            let logits = self.model.forward(&input_tensor, current_pos)?;
            let logits = logits.squeeze(0)?;
            let logits = logits.to_dtype(DType::F32)?;

            next_token = logits_processor.sample(&logits)?;

            if next_token == eos_token_id {
                break;
            }

            generated_tokens.push(next_token);
            current_pos += 1;
        }

        // Decode generated tokens
        let output = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow!("Decoding error: {}", e))?;

        Ok(output)
    }

    fn token_count(&self, text: &str) -> Result<usize> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;
        Ok(encoding.get_ids().len())
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn max_context_length(&self) -> usize {
        self.context_length
    }
}
