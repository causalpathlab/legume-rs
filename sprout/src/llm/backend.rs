use anyhow::Result;

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub temperature: f32,
    pub max_tokens: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            max_tokens: 1024,
            top_p: 0.9,
            repeat_penalty: 1.1,
        }
    }
}

/// Response from an LLM
#[derive(Debug)]
pub enum LlmResponse {
    /// Plain text response
    Text(String),
    /// Tool call request
    ToolCall(ToolCallRequest),
    /// Mixed response with text and tool calls
    Mixed {
        text: String,
        tool_calls: Vec<ToolCallRequest>,
    },
}

/// A request to call a tool
#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    pub tool_name: String,
    pub parameters: serde_json::Value,
}

/// Trait for LLM backends
pub trait LlmBackend: Send + Sync {
    /// Generate text from a prompt
    fn generate(&mut self, prompt: &str, config: &GenerateConfig) -> Result<String>;

    /// Count tokens in text
    fn token_count(&self, text: &str) -> Result<usize>;

    /// Get the model name/identifier
    fn model_name(&self) -> &str;

    /// Get the maximum context length
    fn max_context_length(&self) -> usize;
}
