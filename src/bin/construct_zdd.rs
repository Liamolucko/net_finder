use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

use clap::Parser;
use net_finder::{Cuboid, Zdd};
use postcard::ser_flavors::Flavor;

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
    let zdd = Zdd::construct(cuboid);

    // Write the ZDD out to disk.
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "zdds", &cuboid.to_string()]
        .into_iter()
        .collect();
    fs::create_dir_all(path.parent().unwrap())?;
    let file = BufWriter::new(File::create(path)?);
    postcard::serialize_with_flavor(&zdd, FileFlavor { file })?;
    Ok(())
}
