use anyhow::Result;
use clap::Parser;
use sprout::cli::repl::ChatRepl;
use sprout::llm::candle_llm::CandleLlm;
use sprout::tools::data_beans_tools::register_data_beans_tools;
use sprout::tools::registry::ToolRegistry;
use sprout::{ChatSession, SessionConfig};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "sprout",
    version,
    about = "Interactive LLM chat interface for genomics data analysis",
    long_about = "Sprout provides a natural language interface to data-beans CLI commands.\n\
                  Use local LLMs (Llama, DeepSeek, Mistral) to analyze single-cell genomics data."
)]
struct Args {
    /// Path to the GGUF model file
    #[arg(short, long, required = true)]
    model: PathBuf,

    /// Path to the tokenizer.json file (defaults to same directory as model)
    #[arg(short, long)]
    tokenizer: Option<PathBuf>,

    /// Context length for the model
    #[arg(long, default_value_t = 4096)]
    context_length: usize,

    /// Temperature for sampling (0.0 = deterministic, 1.0 = creative)
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Maximum tokens to generate per response
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,

    /// Use CPU only (disable GPU acceleration)
    #[arg(long, default_value_t = false)]
    cpu: bool,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    // Determine tokenizer path
    let tokenizer_path = args.tokenizer.unwrap_or_else(|| {
        args.model
            .parent()
            .unwrap_or(&args.model)
            .join("tokenizer.json")
    });

    println!("Sprout - Interactive LLM Chat for Genomics");
    println!("==========================================");
    println!("Loading model: {}", args.model.display());
    println!("Tokenizer: {}", tokenizer_path.display());

    // Initialize LLM backend
    let llm = CandleLlm::from_gguf(&args.model, &tokenizer_path, args.cpu, args.context_length)?;

    // Initialize tool registry with data-beans tools
    let mut registry = ToolRegistry::new();
    register_data_beans_tools(&mut registry);

    println!("Registered {} tools", registry.len());

    // Create session config
    let config = SessionConfig {
        temperature: args.temperature,
        max_output_tokens: args.max_tokens,
        ..Default::default()
    };

    // Create chat session
    let session = ChatSession::new(Box::new(llm), registry, config);

    // Run the REPL
    let mut repl = ChatRepl::new(session);
    repl.run()
}
