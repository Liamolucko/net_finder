use std::env;
use std::time::Instant;

use anyhow::bail;
use net_finder::{Cuboid, NetFinder};

fn main() -> anyhow::Result<()> {
    let args = env::args()
        .into_iter()
        .skip(1)
        .map(|arg| arg.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    if args.len() % 3 != 0 {
        bail!(
            "expected sets of 3 arguments representing the width, height and depth of the cuboids."
        );
    }
    let cuboids = args
        .chunks(3)
        .map(|chunk| Cuboid::new(chunk[0], chunk[1], chunk[2]))
        .collect();
    let mut count = 0;
    let start = Instant::now();
    for net in NetFinder::new(cuboids)? {
        println!("{:?}:", start.elapsed());
        println!("{net}");
        println!();
        count += 1;
    }
    println!("Number of nets: {count}");
    Ok(())
}
