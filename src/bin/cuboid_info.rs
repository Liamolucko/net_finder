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

    match *cuboids.as_slice() {
        [] => run([], verbose),
        [a] => run([a], verbose),
        [a, b] => run([a, b], verbose),
        [a, b, c] => run([a, b, c], verbose),
        _ => panic!("only up to 3 cuboids supported"),
    }
}

fn run<const CUBOIDS: usize>(cuboids: [Cuboid; CUBOIDS], verbose: bool) {
    let square_caches = cuboids.map(SquareCache::new);
    let (fixed_cuboid, fixed_class, equivalence_classes) = equivalence_classes(&square_caches);
    println!(
        "Fixed class: {} on {} cuboid",
        fixed_class.index(),
        cuboids[fixed_cuboid]
    );
    println!("Equivalence classes:");
    for (i, class) in equivalence_classes.iter().enumerate() {
        print!("Class {}: {} members", i + 1, class.len());
        if verbose {
            println!(":");
            for mapping in class {
                println!("  {mapping}");
            }
        } else {
            println!();
        }
    }
}
