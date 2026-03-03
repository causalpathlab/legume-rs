//! Sprout: Interactive LLM chat interface for genomics data analysis
//!
//! This crate provides a chat interface that uses local LLMs to invoke
//! data-beans CLI routines via natural language commands.

pub mod chat;
pub mod cli;
pub mod llm;
pub mod tools;

pub use chat::session::ChatSession;
pub use chat::session::SessionConfig;
pub use llm::backend::LlmBackend;
pub use llm::backend::LlmResponse;
pub use tools::registry::ToolRegistry;
