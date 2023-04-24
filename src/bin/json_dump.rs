//! Dumps all the yielded nets stored in one of the saved state files.

use std::{fs, io, path::PathBuf};

use clap::Parser;
use net_finder::State;

#[derive(Parser)]
struct Args {
    path: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let Args { path } = Args::parse();
    let bytes = fs::read(path)?;
    let state: State = postcard::from_bytes(&bytes)?;

    let mut serialized_nets = state
        .yielded_nets
        .into_iter()
        .map(|net| {
            // Convert the nets into a slightly nicer array-of-rows format instead of the
            // width + flat array format.
            // We also canonicalise and later sort all the nets so that everything's nice
            // and deterministic.
            net.canon()
                .rows()
                .map(|row| row.to_owned())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    serialized_nets.sort_unstable();
    serde_json::to_writer(io::stdout(), &serialized_nets)?;

    Ok(())
}
