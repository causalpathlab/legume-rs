use std::collections::HashMap;

/// Type of a parameter
#[derive(Debug, Clone)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    StringArray,
    IntegerArray,
    FilePath,
    Enum(Vec<String>),
}

impl std::fmt::Display for ParameterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParameterType::String => write!(f, "string"),
            ParameterType::Integer => write!(f, "integer"),
            ParameterType::Float => write!(f, "float"),
            ParameterType::Boolean => write!(f, "boolean"),
            ParameterType::StringArray => write!(f, "string[]"),
            ParameterType::IntegerArray => write!(f, "integer[]"),
            ParameterType::FilePath => write!(f, "file_path"),
            ParameterType::Enum(values) => write!(f, "enum[{}]", values.join("|")),
        }
    }
}

/// Definition of a tool parameter
#[derive(Debug, Clone)]
pub struct ToolParameter {
    pub name: String,
    pub description: String,
    pub param_type: ParameterType,
    pub required: bool,
    pub default: Option<serde_json::Value>,
}

/// Handler function type for tools
pub type ToolHandlerFn = Box<dyn Fn(&serde_json::Value) -> anyhow::Result<String> + Send + Sync>;

/// Definition of a tool
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
    pub handler: ToolHandlerFn,
}

impl std::fmt::Debug for ToolDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolDefinition")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .finish()
    }
}

/// Registry of available tools
pub struct ToolRegistry {
    tools: HashMap<String, ToolDefinition>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a new tool
    pub fn register(&mut self, tool: ToolDefinition) {
        self.tools.insert(tool.name.clone(), tool);
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }

    /// List all tool names
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Get number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Format tools for inclusion in a prompt
    pub fn format_tools_for_prompt(&self) -> String {
        let mut lines = Vec::new();

        for tool in self.tools.values() {
            lines.push(format!("### {}", tool.name));
            lines.push(format!("{}", tool.description));
            lines.push("Parameters:".to_string());

            for param in &tool.parameters {
                let required_str = if param.required { "(required)" } else { "(optional)" };
                let default_str = param
                    .default
                    .as_ref()
                    .map(|v| format!(", default: {}", v))
                    .unwrap_or_default();

                lines.push(format!(
                    "  - {}: {} {} - {}{}",
                    param.name, param.param_type, required_str, param.description, default_str
                ));
            }
            lines.push(String::new());
        }

        lines.join("\n")
    }

    /// Format a brief list of tools
    pub fn format_tools_list(&self) -> String {
        let mut lines = Vec::new();
        for tool in self.tools.values() {
            lines.push(format!("  {} - {}", tool.name, tool.description));
        }
        lines.join("\n")
    }
}
