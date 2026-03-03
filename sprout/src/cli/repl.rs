use crate::chat::session::ChatSession;
use anyhow::Result;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

const PROMPT: &str = "> ";

/// Interactive REPL for chat interface
pub struct ChatRepl {
    session: ChatSession,
    editor: DefaultEditor,
}

impl ChatRepl {
    pub fn new(session: ChatSession) -> Self {
        let editor = DefaultEditor::new().expect("Failed to create editor");
        Self { session, editor }
    }

    /// Run the REPL loop
    pub fn run(&mut self) -> Result<()> {
        println!("\nType /help for available commands, or start chatting!\n");

        loop {
            let readline = self.editor.readline(PROMPT);

            match readline {
                Ok(line) => {
                    let line = line.trim();

                    if line.is_empty() {
                        continue;
                    }

                    // Add to history
                    let _ = self.editor.add_history_entry(line);

                    // Handle commands
                    if line.starts_with('/') {
                        match self.handle_command(line) {
                            Ok(should_continue) => {
                                if !should_continue {
                                    break;
                                }
                            }
                            Err(e) => {
                                println!("Error: {}\n", e);
                            }
                        }
                        continue;
                    }

                    // Send message to chat
                    match self.session.send_message(line) {
                        Ok(response) => {
                            // Print text response
                            if !response.text.is_empty() {
                                println!("\n{}\n", response.text);
                            }

                            // Print tool call results
                            for tool_call in &response.tool_calls {
                                let status = if tool_call.success { "OK" } else { "FAILED" };
                                println!("[Tool: {} - {}]", tool_call.tool_name, status);
                                println!("{}\n", tool_call.output);
                            }
                        }
                        Err(e) => {
                            println!("Error: {}\n", e);
                        }
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    println!("Goodbye!");
                    break;
                }
                Err(err) => {
                    println!("Error: {:?}", err);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle slash commands
    fn handle_command(&mut self, command: &str) -> Result<bool> {
        let parts: Vec<&str> = command.splitn(2, ' ').collect();
        let cmd = parts[0];
        let _args = parts.get(1).unwrap_or(&"");

        match cmd {
            "/help" | "/h" => {
                self.print_help();
            }
            "/quit" | "/q" | "/exit" => {
                println!("Goodbye!");
                return Ok(false);
            }
            "/clear" | "/c" => {
                self.session.clear_history();
                println!("Conversation history cleared.\n");
            }
            "/tools" | "/t" => {
                self.print_tools();
            }
            _ => {
                println!("Unknown command: {}", cmd);
                println!("Type /help for available commands.\n");
            }
        }

        Ok(true)
    }

    fn print_help(&self) {
        println!("\nAvailable commands:");
        println!("  /help, /h     - Show this help message");
        println!("  /tools, /t    - List available tools");
        println!("  /clear, /c    - Clear conversation history");
        println!("  /quit, /q     - Exit the chat");
        println!();
        println!("You can also just type natural language to interact with your data.");
        println!("Example: \"Show info about my_data.zarr\"\n");
    }

    fn print_tools(&self) {
        println!("\nAvailable tools:");
        println!("{}\n", self.session.tool_registry().format_tools_list());
    }
}
