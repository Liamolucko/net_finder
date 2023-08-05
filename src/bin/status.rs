//! Prints out the status of a state file.

use std::{fs::File, io::BufReader, path::PathBuf, time::Duration};

use clap::Parser;
use serde::{de::IgnoredAny, Deserialize};

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
    println!("Runtime: {:.3?}", state.prior_search_time);
    println!("{} finders", state.finders.len());

    Ok(())
}
