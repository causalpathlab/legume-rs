use crate::tools::registry::ToolDefinition;
use anyhow::Result;

/// Executes tool calls
pub struct ToolExecutor {
    // Future: add safety config, allowed paths, etc.
}

impl Default for ToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolExecutor {
    pub fn new() -> Self {
        Self {}
    }

    /// Execute a tool with given parameters
    pub fn execute(&self, tool: &ToolDefinition, params: &serde_json::Value) -> Result<String> {
        log::debug!("Executing tool '{}' with params: {}", tool.name, params);

        // Call the tool handler
        let result = (tool.handler)(params)?;

        log::debug!("Tool '{}' completed successfully", tool.name);
        Ok(result)
    }
}

/// Helper trait for extracting parameters from JSON
pub trait ParamExtractor {
    fn get_string(&self, key: &str) -> Result<String>;
    fn get_string_or(&self, key: &str, default: &str) -> String;
    fn get_usize(&self, key: &str) -> Result<usize>;
    fn get_usize_or(&self, key: &str, default: usize) -> usize;
    fn get_f32(&self, key: &str) -> Result<f32>;
    fn get_f32_or(&self, key: &str, default: f32) -> f32;
    fn get_bool(&self, key: &str) -> Result<bool>;
    fn get_bool_or(&self, key: &str, default: bool) -> bool;
    fn get_string_array(&self, key: &str) -> Result<Vec<String>>;
    fn get_usize_array(&self, key: &str) -> Result<Vec<usize>>;
}

impl ParamExtractor for serde_json::Value {
    fn get_string(&self, key: &str) -> Result<String> {
        self.get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("Missing required string parameter: {}", key))
    }

    fn get_string_or(&self, key: &str, default: &str) -> String {
        self.get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| default.to_string())
    }

    fn get_usize(&self, key: &str) -> Result<usize> {
        self.get(key)
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| anyhow::anyhow!("Missing required integer parameter: {}", key))
    }

    fn get_usize_or(&self, key: &str, default: usize) -> usize {
        self.get(key)
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(default)
    }

    fn get_f32(&self, key: &str) -> Result<f32> {
        self.get(key)
            .and_then(|v| v.as_f64())
            .map(|n| n as f32)
            .ok_or_else(|| anyhow::anyhow!("Missing required float parameter: {}", key))
    }

    fn get_f32_or(&self, key: &str, default: f32) -> f32 {
        self.get(key)
            .and_then(|v| v.as_f64())
            .map(|n| n as f32)
            .unwrap_or(default)
    }

    fn get_bool(&self, key: &str) -> Result<bool> {
        self.get(key)
            .and_then(|v| v.as_bool())
            .ok_or_else(|| anyhow::anyhow!("Missing required boolean parameter: {}", key))
    }

    fn get_bool_or(&self, key: &str, default: bool) -> bool {
        self.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
    }

    fn get_string_array(&self, key: &str) -> Result<Vec<String>> {
        self.get(key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .ok_or_else(|| anyhow::anyhow!("Missing required string array parameter: {}", key))
    }

    fn get_usize_array(&self, key: &str) -> Result<Vec<usize>> {
        self.get(key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .ok_or_else(|| anyhow::anyhow!("Missing required integer array parameter: {}", key))
    }
}
