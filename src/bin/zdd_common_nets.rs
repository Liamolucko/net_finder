//! Finds the common developments of two cuboids from their ZDDs (well, from
//! their already-serialized nets that originated from their ZDDs.)

use std::{fs, path::PathBuf};

use anyhow::bail;
use clap::Parser;
use indicatif::{MultiProgress, ParallelProgressIterator, ProgressBar, ProgressStyle};
use net_finder::{Cuboid, Net, Zdd};
use rayon::prelude::ParallelIterator;
use rustc_hash::FxHashSet;

#[derive(Parser)]
struct Options {
    cuboids: Vec<Cuboid>,
}

fn main() -> anyhow::Result<()> {
    let Options { cuboids } = Options::parse();

    // Load the ZDD for the first cuboid.
    if cuboids.len() < 1 {
        bail!("expected at least 1 cuboid");
    }
    let first_cuboid = cuboids[0];
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "zdds",
        &first_cuboid.to_string(),
    ]
    .into_iter()
    .collect();
    let bytes = fs::read(path)?;
    let zdd: Zdd = postcard::from_bytes(&bytes)?;
    drop(bytes);

    let progress = MultiProgress::new();
    let style = ProgressStyle::with_template("{prefix}: {human_pos} {spinner}").unwrap();
    let zdd_progress = ProgressBar::new_spinner()
        .with_prefix("nets read")
        .with_style(style.clone());
    let found_progress = ProgressBar::new_spinner()
        .with_prefix("common nets found")
        .with_style(style.clone());
    progress.add(zdd_progress.clone());
    progress.add(found_progress.clone());

    // Then go through the ZDD and filter out the ones which work for all the
    // cuboids.
    let common_nets: FxHashSet<Net> = zdd
        .par_nets()
        .progress_with(zdd_progress)
        .filter(|net| {
            cuboids
                .iter()
                .copied()
                .all(|cuboid| net.color(cuboid).is_some())
        })
        .progress_with(found_progress)
        .collect();

    // Write them to disk.
    let filename = cuboids
        .iter()
        .map(|cuboid| cuboid.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "common_nets", &filename]
        .into_iter()
        .collect();
    let bytes = postcard::to_stdvec(&common_nets)?;
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, bytes)?;

    for (i, net) in common_nets.into_iter().enumerate() {
        println!("Net #{i}:\n{}", net.color(cuboids[0]).unwrap());
    }
    Ok(())
}
