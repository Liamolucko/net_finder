use std::env;

use anyhow::bail;
use net_finder::Cuboid;

fn main() -> anyhow::Result<()> {
    let &[width, depth, height] = env::args().into_iter().skip(1).map(|arg| arg.parse::<usize>()).collect::<Result<Vec<_>, _>>()?.as_slice() else {
        bail!("Expected 3 args, the width, depth and height of the cuboid.");
    };
    let net = net_finder::find_nets(Cuboid::new(width, depth, height));
    println!("{net}");
    Ok(())
}
