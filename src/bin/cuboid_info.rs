use clap::Parser;
use net_finder::{equivalence_classes, Cuboid, SquareCache, Zdd};

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

    let square_caches: Vec<_> = cuboids
        .iter()
        .map(|&cuboid| SquareCache::new(cuboid))
        .collect();
    let equivalence_classes = equivalence_classes(&cuboids, &square_caches);
    println!("Equivalence classes:");
    for (i, class) in equivalence_classes.iter().enumerate() {
        println!(
            "Class {}: {} canon members",
            i + 1,
            class.canon_mappings().count()
        );
    }
}
