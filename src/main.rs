use std::time::Instant;

use chrono::{DateTime, Local};
use clap::Parser;
use net_finder::{find_nets, resume, Cuboid, Solution, MAX_CUBOIDS};

#[derive(Parser)]
struct Options {
    #[arg(long)]
    resume: bool,
    cuboids: Vec<Cuboid>,
}

fn main() -> anyhow::Result<()> {
    let options = Options::parse();
    assert!(options.cuboids.len() <= MAX_CUBOIDS.into());

    let mut count = 0;
    let start = Instant::now();
    let callback = |solution: Solution| {
        count += 1;
        println!(
            "#{count} after {:.3?} ({}):",
            solution.search_time,
            DateTime::<Local>::from(solution.time).format("at %r on %e %b")
        );
        println!("{solution}");
    };
    if options.resume {
        resume(&options.cuboids)?.for_each(callback);
    } else {
        find_nets(&options.cuboids)?.for_each(callback);
    }
    println!("Number of nets: {count} (took {:?})", start.elapsed());
    Ok(())
}
