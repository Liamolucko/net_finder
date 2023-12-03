//! Dumps all the yielded nets stored in one of the saved state files.

use std::fs::File;
use std::io::{self, BufReader};
use std::path::PathBuf;

use clap::Parser;
use net_finder::Solution;
use serde::{Deserialize, Serialize};

#[derive(Parser)]
struct Args {
    path: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct State {
    // We only need this field, not the whole thing.
    solutions: Vec<Solution>,
}

fn main() -> anyhow::Result<()> {
    let Args { path } = Args::parse();
    let file = File::open(path)?;
    let state: State = serde_json::from_reader(BufReader::new(file))?;

    let mut serialized_nets = state
        .solutions
        .into_iter()
        .map(|solution| {
            // Convert the nets into a slightly nicer array-of-rows format instead of the
            // width + flat array format.
            // We also canonicalise and later sort all the nets so that everything's nice
            // and deterministic.
            solution
                .net
                .rows()
                .map(|row| row.to_owned())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    serialized_nets.sort_unstable();
    serde_json::to_writer(io::stdout(), &serialized_nets)?;

    Ok(())
}
