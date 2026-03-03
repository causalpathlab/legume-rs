use crate::chat::history::{ConversationHistory, Role};
use crate::llm::backend::{GenerateConfig, LlmBackend};
use crate::tools::executor::ToolExecutor;
use crate::tools::registry::ToolRegistry;
use anyhow::Result;

/// Configuration for a chat session
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub max_history_messages: usize,
    pub system_prompt: String,
    pub temperature: f32,
    pub max_output_tokens: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_history_messages: 20,
            system_prompt: default_system_prompt(),
            temperature: 0.7,
            max_output_tokens: 1024,
        }
    }
}

fn default_system_prompt() -> String {
    r#"You are a helpful assistant with access to tools for analyzing genomics data files.

To use a tool, respond with a JSON block like this:
```tool
{"tool": "show_matrix_info", "parameters": {"data_file": "/path/to/file.zarr"}}
```

Available tools:
- show_matrix_info: Get info about a matrix file (rows, columns, non-zeros)
- list_row_names: List gene/feature names from a matrix
- list_column_names: List cell/sample names from a matrix

When users ask about files, use the appropriate tool."#
        .to_string()
}

/// Response from a chat interaction
#[derive(Debug)]
pub struct ChatResponse {
    pub text: String,
    pub tool_calls: Vec<ToolCallResult>,
}

/// Result of a tool call
#[derive(Debug)]
pub struct ToolCallResult {
    pub tool_name: String,
    pub success: bool,
    pub output: String,
}

/// Main chat session managing conversation state
pub struct ChatSession {
    history: ConversationHistory,
    llm: Box<dyn LlmBackend>,
    tool_registry: ToolRegistry,
    tool_executor: ToolExecutor,
    config: SessionConfig,
}

impl ChatSession {
    pub fn new(llm: Box<dyn LlmBackend>, tool_registry: ToolRegistry, config: SessionConfig) -> Self {
        let mut history = ConversationHistory::new(config.max_history_messages);
        history.add_message(Role::System, config.system_prompt.clone());

        let tool_executor = ToolExecutor::new();

        Self {
            history,
            llm,
            tool_registry,
            tool_executor,
            config,
        }
    }

    /// Send a message and get a response
    pub fn send_message(&mut self, user_input: &str) -> Result<ChatResponse> {
        // Add user message to history
        self.history.add_message(Role::User, user_input.to_string());

        // Build prompt - use TinyLlama/Zephyr chat format
        let prompt = format!(
            "{}\n<|assistant|>\n",
            self.history.format_for_prompt()
        );

        // Generate response
        let gen_config = GenerateConfig {
            temperature: self.config.temperature,
            max_tokens: self.config.max_output_tokens,
            ..Default::default()
        };

        let response = self.llm.generate(&prompt, &gen_config)?;

        // Parse response for tool calls
        let (text, tool_calls) = self.process_response(&response)?;

        // Add assistant response to history
        self.history.add_message(Role::Assistant, response);

        Ok(ChatResponse { text, tool_calls })
    }

    /// Process response to extract text and execute any tool calls
    fn process_response(&mut self, response: &str) -> Result<(String, Vec<ToolCallResult>)> {
        let mut tool_calls = Vec::new();
        let mut text_parts = Vec::new();
        let mut remaining = response;

        // Look for ```tool blocks
        while let Some(start) = remaining.find("```tool") {
            // Add text before the tool block
            if start > 0 {
                text_parts.push(remaining[..start].trim());
            }

            // Find end of tool block
            let block_start = start + 7; // len of "```tool"
            if let Some(end) = remaining[block_start..].find("```") {
                let tool_json = remaining[block_start..block_start + end].trim();

                // Try to parse and execute tool call
                match self.execute_tool_call(tool_json) {
                    Ok(result) => {
                        // Add tool result to history
                        self.history.add_message(
                            Role::Tool,
                            format!("[{}]: {}", result.tool_name, result.output),
                        );
                        tool_calls.push(result);
                    }
                    Err(e) => {
                        tool_calls.push(ToolCallResult {
                            tool_name: "unknown".to_string(),
                            success: false,
                            output: format!("Failed to parse tool call: {}", e),
                        });
                    }
                }

                remaining = &remaining[block_start + end + 3..];
            } else {
                // No closing ```, treat rest as text
                text_parts.push(&remaining[start..]);
                break;
            }
        }

        // Add any remaining text
        if !remaining.is_empty() {
            text_parts.push(remaining.trim());
        }

        let text = text_parts.join("\n").trim().to_string();
        Ok((text, tool_calls))
    }

    /// Execute a single tool call from JSON
    fn execute_tool_call(&mut self, json_str: &str) -> Result<ToolCallResult> {
        let call: serde_json::Value = serde_json::from_str(json_str)?;

        let tool_name = call
            .get("tool")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'tool' field"))?;

        let params = call
            .get("parameters")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        // Look up tool in registry
        if let Some(tool) = self.tool_registry.get(tool_name) {
            match self.tool_executor.execute(tool, &params) {
                Ok(output) => Ok(ToolCallResult {
                    tool_name: tool_name.to_string(),
                    success: true,
                    output,
                }),
                Err(e) => Ok(ToolCallResult {
                    tool_name: tool_name.to_string(),
                    success: false,
                    output: format!("Error: {}", e),
                }),
            }
        } else {
            Ok(ToolCallResult {
                tool_name: tool_name.to_string(),
                success: false,
                output: format!("Unknown tool: {}", tool_name),
            })
        }
    }

    /// Clear conversation history (keeps system prompt)
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.history
            .add_message(Role::System, self.config.system_prompt.clone());
    }

    /// Get the tool registry (for listing available tools)
    pub fn tool_registry(&self) -> &ToolRegistry {
        &self.tool_registry
    }
}
