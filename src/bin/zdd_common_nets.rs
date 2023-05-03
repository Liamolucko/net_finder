//! Finds the common developments of two cuboids from their ZDDs (well, from
//! their already-serialized nets that originated from their ZDDs.)

use std::{fs, path::PathBuf};

use clap::Parser;
use net_finder::{Cuboid, Net};
use rustc_hash::FxHashSet;

#[derive(Parser)]
struct Options {
    cuboids: Vec<Cuboid>,
}

fn main() -> anyhow::Result<()> {
    let Options { cuboids } = Options::parse();
    let mut common_nets = FxHashSet::default();
    for cuboid in cuboids.iter() {
        let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "nets", &cuboid.to_string()]
            .into_iter()
            .collect();
        let bytes = fs::read(path)?;
        let nets: FxHashSet<Net> = postcard::from_bytes(&bytes)?;
        if common_nets.is_empty() {
            common_nets = nets;
        } else {
            common_nets.retain(|net| nets.contains(net));
        }
    }
    for (i, net) in common_nets.into_iter().enumerate() {
        println!("Net #{i}:\n{}", net.color(cuboids[0]).unwrap());
    }
    Ok(())
}
