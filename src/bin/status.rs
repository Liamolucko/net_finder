//! Prints out the status of a state file.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use indicatif::FormattedDuration;
use serde::de::IgnoredAny;
use serde::Deserialize;

#[derive(Parser)]
struct Args {
    path: PathBuf,
}

#[derive(Deserialize)]
struct State {
    prior_search_time: Duration,
    finders: Vec<IgnoredAny>,
}

fn main() -> anyhow::Result<()> {
    let Args { path } = Args::parse();
    let file = File::open(path)?;
    let state: State = serde_json::from_reader(BufReader::new(file))?;
    println!("Runtime: {}", FormattedDuration(state.prior_search_time));
    println!("{} finders", state.finders.len());

    Ok(())
}
