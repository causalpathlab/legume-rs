//! Interactive user input utilities for command-line prompts
//!
//! This module provides reusable functions for interactive CLI operations
//! including confirmations, option selection, and numeric input.

#![allow(dead_code)] // Utility functions for future use

use std::io::{self, Write};

/// User action after viewing histogram or other interactive prompts
#[derive(Debug, Clone)]
pub enum UserAction {
    Proceed,
    AdjustCutoffs(usize, usize),
    Cancel,
}

/// Prompt user for action in interactive mode after showing histogram
pub fn prompt_user_action(
    _row_nnz: &[f32],
    _col_nnz: &[f32],
    current_row_cutoff: usize,
    current_col_cutoff: usize,
) -> anyhow::Result<UserAction> {
    println!("\nOptions:");
    println!("  [p] Proceed with current cutoffs");
    println!("  [a] Adjust cutoffs");
    println!("  [c] Cancel");
    print!("\nChoose an option (p/a/c): ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let choice = input.trim().to_lowercase();

    match choice.as_str() {
        "p" | "proceed" | "y" | "yes" => Ok(UserAction::Proceed),
        "c" | "cancel" | "n" | "no" => Ok(UserAction::Cancel),
        "a" | "adjust" => {
            let new_row_cutoff = prompt_cutoff_value("row", current_row_cutoff)?;
            let new_col_cutoff = prompt_cutoff_value("column", current_col_cutoff)?;
            Ok(UserAction::AdjustCutoffs(new_row_cutoff, new_col_cutoff))
        }
        _ => {
            println!("Invalid choice. Cancelling operation.");
            Ok(UserAction::Cancel)
        }
    }
}

/// Prompt user for a single cutoff value
pub fn prompt_cutoff_value(label: &str, current: usize) -> anyhow::Result<usize> {
    print!("\nEnter new {} nnz cutoff (current: {}): ", label, current);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    let value = input.trim().parse::<usize>().unwrap_or(current);
    Ok(value)
}

/// Simple yes/no confirmation prompt
pub fn confirm(message: &str) -> anyhow::Result<bool> {
    print!("{} (y/n): ", message);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let choice = input.trim().to_lowercase();

    Ok(matches!(choice.as_str(), "y" | "yes"))
}

/// Prompt user to select from a list of options
pub fn select_option(prompt: &str, options: &[&str]) -> anyhow::Result<usize> {
    println!("\n{}", prompt);
    for (i, opt) in options.iter().enumerate() {
        println!("  [{}] {}", i + 1, opt);
    }
    print!("\nSelect option (1-{}): ", options.len());
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    if let Ok(choice) = input.trim().parse::<usize>() {
        if choice > 0 && choice <= options.len() {
            return Ok(choice - 1);
        }
    }

    Err(anyhow::anyhow!("Invalid selection"))
}

/// Prompt for a numeric value with validation
pub fn prompt_number<T>(prompt: &str, default: T) -> anyhow::Result<T>
where
    T: std::str::FromStr + std::fmt::Display + Copy,
{
    print!("{} [default: {}]: ", prompt, default);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    if input.trim().is_empty() {
        return Ok(default);
    }

    input.trim().parse::<T>()
        .map_err(|_| anyhow::anyhow!("Invalid number format"))
}

/// Read a line of text from user
pub fn read_line(prompt: &str) -> anyhow::Result<String> {
    print!("{}: ", prompt);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}
