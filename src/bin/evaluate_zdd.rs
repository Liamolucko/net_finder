//! Evaluates all of the nets that a ZDD represents.
//!
//! This is what `Zdd::size` is supposed to do, but it doesn't work.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::{fs, io::BufWriter};

use clap::Parser;
use indicatif::HumanBytes;
use net_finder::{Cuboid, Net, Zdd};
use postcard::ser_flavors::Flavor;
use rayon::prelude::*;
use rustc_hash::FxHashSet;

#[derive(Parser)]
struct Options {
    cuboid: Cuboid,
}

/// A postcard `Flavor` that writes to a file.
struct FileFlavor {
    file: BufWriter<File>,
}

impl Flavor for FileFlavor {
    type Output = ();

    fn try_push(&mut self, data: u8) -> postcard::Result<()> {
        self.file
            .write_all(&[data])
            .map_err(|_| postcard::Error::SerdeSerCustom)
    }

    fn try_extend(&mut self, data: &[u8]) -> postcard::Result<()> {
        self.file
            .write_all(data)
            .map_err(|_| postcard::Error::SerdeSerCustom)
    }

    fn finalize(self) -> postcard::Result<Self::Output> {
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let Options { cuboid } = Options::parse();
    let in_path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "zdds", &cuboid.to_string()]
        .into_iter()
        .collect();
    let bytes = fs::read(in_path)?;
    let zdd: Zdd = postcard::from_bytes(&bytes)?;
    // No need to keep this 3GB binary blob around.
    drop(bytes);
    println!("ZDD takes up {}.", HumanBytes(zdd.heap_size() as u64));

    let nets: FxHashSet<Net> = zdd.par_nets().collect();
    let nets: Vec<Net> = nets
        .into_par_iter()
        .filter(|net| net.color(cuboid).is_some())
        .collect();
    println!("ZDD contains {} nets.", nets.len());

    // Write the nets to disk.
    let out_path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "nets", &cuboid.to_string()]
        .into_iter()
        .collect();
    fs::create_dir_all(out_path.parent().unwrap())?;
    let file = BufWriter::new(File::create(out_path)?);
    postcard::serialize_with_flavor(&nets, FileFlavor { file })?;

    Ok(())
}
