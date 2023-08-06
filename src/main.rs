use std::time::Duration;

use chrono::{DateTime, Local};
use clap::Parser;
use indicatif::ProgressBar;
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

    let state = if options.resume {
        Some(read_state(&options.cuboids)?)
    } else {
        None
    };

    let prior_search_time = state
        .as_ref()
        .map_or(Duration::ZERO, |state| state.prior_search_time);
    // Make a progress bar for `find_nets` to do with as it will.
    let progress = ProgressBar::hidden().with_elapsed(prior_search_time);

    let mut count = 0;
    let callback = |solution: Solution| {
        progress.suspend(|| {
            count += 1;
            println!(
                "#{count} after {:.3?} ({}):",
                solution.search_time,
                DateTime::<Local>::from(solution.time).format("at %r on %e %b")
            );
            println!("{solution}");
        });
    };
    if let Some(state) = state {
        resume(state, progress.clone()).for_each(callback);
    } else {
        find_nets(&options.cuboids, progress.clone())?.for_each(callback);
    }

    progress.finish_and_clear();
    println!("Number of nets: {count} (took {:?})", progress.elapsed());
    Ok(())
}
