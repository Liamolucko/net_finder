use std::time::Duration;

use anyhow::bail;
use chrono::{DateTime, Local};
use clap::Parser;
use indicatif::ProgressBar;
use net_finder::{read_state, Cuboid, Solution};

#[derive(Parser)]
struct Options {
    #[arg(long)]
    resume: bool,
    cuboids: Vec<Cuboid>,
}

fn main() -> anyhow::Result<()> {
    let options = Options::parse();
    match options.cuboids.as_slice() {
        &[] => bail!("must specify at least 1 cuboid"),
        &[a] => run([a], options.resume),
        &[a, b] => run([a, b], options.resume),
        &[a, b, c] => run([a, b, c], options.resume),
        _ => bail!("only up to 3 cuboids are currently supported"),
    }
}

fn run<const CUBOIDS: usize>(cuboids: [Cuboid; CUBOIDS], resume: bool) -> anyhow::Result<()> {
    let state = if resume {
        Some(read_state(cuboids)?)
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
        net_finder::resume(state, progress.clone()).for_each(callback);
    } else {
        net_finder::find_nets(cuboids, progress.clone())?.for_each(callback);
    }

    progress.finish_and_clear();
    println!("Number of nets: {count} (took {:?})", progress.elapsed());
    Ok(())
}
