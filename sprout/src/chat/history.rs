use serde::{Deserialize, Serialize};

/// A single message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// The role of a message sender
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

/// Conversation history management
#[derive(Debug, Default)]
pub struct ConversationHistory {
    messages: Vec<Message>,
    max_messages: usize,
}

impl ConversationHistory {
    pub fn new(max_messages: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_messages,
        }
    }

    pub fn add_message(&mut self, role: Role, content: String) {
        self.messages.push(Message { role, content });

        // Trim old messages if exceeding limit (keep system message if present)
        while self.messages.len() > self.max_messages {
            // Find first non-system message to remove
            if let Some(idx) = self.messages.iter().position(|m| m.role != Role::System) {
                self.messages.remove(idx);
            } else {
                break;
            }
        }
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Format history using TinyLlama/Zephyr chat template
    pub fn format_for_prompt(&self) -> String {
        self.messages
            .iter()
            .map(|m| format!("<|{}|>\n{}</s>", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}
