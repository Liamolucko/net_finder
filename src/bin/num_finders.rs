//! Prints out how many active `NetFinder`s there are in a state file.

use std::{fs::File, io::BufReader, path::PathBuf};

use clap::Parser;
use net_finder::State;

#[derive(Parser)]
struct Args {
    path: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let Args { path } = Args::parse();
    let file = File::open(path)?;
    let state: State = serde_json::from_reader(BufReader::new(file))?;
    println!("{} active finders", state.finders.len());

    Ok(())
}
