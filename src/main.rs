use std::time::{Duration, Instant};

use chrono::{DateTime, Local};
use clap::Parser;
use net_finder::{find_nets, read_state, resume, Cuboid, Solution, MAX_CUBOIDS};

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
    let prior_search_time = if options.resume {
        let state = read_state(&options.cuboids)?;
        let prior_search_tiem = state.prior_search_time;
        resume(state).for_each(callback);
        prior_search_tiem
    } else {
        find_nets(&options.cuboids)?.for_each(callback);
        Duration::ZERO
    };
    println!(
        "Number of nets: {count} (took {:?})",
        prior_search_time + start.elapsed()
    );
    Ok(())
}
