use std::env;

use anyhow::bail;
use net_finder::{Cuboid, NetFinder};

fn main() -> anyhow::Result<()> {
    let &[width, depth, height] = env::args().into_iter().skip(1).map(|arg| arg.parse::<usize>()).collect::<Result<Vec<_>, _>>()?.as_slice() else {
        bail!("Expected 3 args, the width, depth and height of the cuboid.");
    };
    let mut count = 0;
    for net in NetFinder::new(Cuboid::new(width, height, depth)) {
        println!("{net}");
        println!();
        count += 1;
    }
    println!("Number of nets: {count}");
    Ok(())
}
