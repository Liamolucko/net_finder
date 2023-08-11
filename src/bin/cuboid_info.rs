use std::collections::HashSet;

use clap::Parser;
use net_finder::{equivalence_classes, Cuboid, SquareCache, Zdd};

#[derive(Parser)]
struct Options {
    cuboids: Vec<Cuboid>,
    /// Enable verbose output (printing the contents of equivalence classes).
    #[arg(short)]
    verbose: bool,
}

fn main() {
    let Options { cuboids, verbose } = Options::parse();
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

    match cuboids.as_slice() {
        &[] => run([], verbose),
        &[a] => run([a], verbose),
        &[a, b] => run([a, b], verbose),
        &[a, b, c] => run([a, b, c], verbose),
        _ => panic!("only up to 3 cuboids supported"),
    }
}

fn run<const CUBOIDS: usize>(cuboids: [Cuboid; CUBOIDS], verbose: bool) {
    let square_caches = cuboids.map(SquareCache::new);
    let equivalence_classes = equivalence_classes(cuboids, &square_caches);
    println!("Equivalence classes:");
    for (i, class) in equivalence_classes.iter().enumerate() {
        print!(
            "Class {}: {} canon members",
            i + 1,
            class.canon_mappings().count(),
        );
        if verbose {
            let canon_mappings: HashSet<_> = class
                .canon_mappings()
                .map(|mapping| mapping.to_data(&square_caches).canon_orientation())
                .collect();
            println!(" ({} rotated): ", canon_mappings.len());
            for mapping in canon_mappings {
                println!("  {mapping}");
            }
        } else {
            println!();
        }
    }
}
