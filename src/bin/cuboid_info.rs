use clap::Parser;
use net_finder::{Cuboid, Zdd};

#[derive(Parser)]
struct Options {
    cuboid: Cuboid,
}

fn main() {
    let Options { cuboid } = Options::parse();
    let (vertices, edges) = Zdd::cuboid_info(cuboid);
    println!("{cuboid} cuboid has {vertices} vertices and {edges} edges");
}
