use std::{fs, path::PathBuf};

use clap::Parser;
use net_finder::{Cuboid, Zdd};

#[derive(Parser)]
struct Options {
    cuboids: Vec<Cuboid>,
}

fn main() -> anyhow::Result<()> {
    let options = Options::parse();
    for cuboid in options.cuboids {
        let zdd = Zdd::construct(cuboid);

        // Write the ZDD out to disk.
        let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "zdds", &cuboid.to_string()]
            .into_iter()
            .collect();
        fs::create_dir_all(path.parent().unwrap())?;
        let bytes = postcard::to_stdvec(&zdd)?;
        fs::write(path, bytes)?;

        let size = zdd.size();
        println!("{size} nets found");
        // for (i, net) in zdd.nets().enumerate() {
        //     print!("net {i} / {size}");
        //     if let Some(colored) = net.color(cuboid) {
        //         println!(":\n{colored}");
        //     } else {
        //         println!(" (broken):\n{net}");
        //     }
        //     println!();
        // }
    }
    Ok(())
}
