use clap::Parser;
use net_finder::{equivalence_classes, Cuboid, Zdd};

#[derive(Parser)]
struct Options {
    cuboids: Vec<Cuboid>,
}

fn main() {
    let Options { cuboids } = Options::parse();
    for cuboid in cuboids.iter().copied() {
        let surface_area = cuboid.surface_area();
        let (vertices, edges) = Zdd::cuboid_info(cuboid);
        println!("{cuboid} cuboid:");
        println!("  surface area = {surface_area}");
        println!("  ZDD info:");
        println!("    vertices = {vertices}");
        println!("    edges = {edges}");
        println!();
    }

    let equivalence_classes = equivalence_classes(&cuboids);
    println!("Equivalence classes:");
    for (i, class) in equivalence_classes.iter().enumerate() {
        println!("Class {}: {} members", i + 1, class.len());
    }
}
