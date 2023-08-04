//! Prints out how many active `NetFinder`s there are in a state file.

use std::{fs::File, io::BufReader, path::PathBuf};

use clap::Parser;
use serde::{de::IgnoredAny, Deserialize};

#[derive(Parser)]
struct Args {
    path: PathBuf,
}

#[derive(Deserialize)]
struct State {
    finders: Vec<IgnoredAny>,
}

fn main() -> anyhow::Result<()> {
    let Args { path } = Args::parse();
    let file = File::open(path)?;
    let state: State = serde_json::from_reader(BufReader::new(file))?;
    println!("{} active finders", state.finders.len());

    Ok(())
}
