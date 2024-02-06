//! The initial home-grown algorithm I came up with.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

use indicatif::ProgressBar;
use net_finder::{Cuboid, Finder, FinderCtx, FinderInfo, Runtime, Solution};

/// A `Runtime` that runs everything on the CPU.
pub struct CpuRuntime<const CUBOIDS: usize> {
    cuboids: [Cuboid; CUBOIDS],
}

impl<const CUBOIDS: usize> CpuRuntime<CUBOIDS> {
    /// Makes a new runtime for running finders on the CPU.
    pub fn new(cuboids: [Cuboid; CUBOIDS]) -> Self {
        Self { cuboids }
    }
}

impl<const CUBOIDS: usize> Runtime<CUBOIDS> for CpuRuntime<CUBOIDS> {
    fn cuboids(&self) -> [Cuboid; CUBOIDS] {
        self.cuboids
    }

    fn run(
        self,
        ctx: &FinderCtx<CUBOIDS>,
        finders: &[FinderInfo<CUBOIDS>],
        solution_tx: &mpsc::Sender<Solution>,
        pause: &AtomicBool,
        progress: Option<&ProgressBar>,
    ) -> anyhow::Result<Vec<FinderInfo<CUBOIDS>>> {
        let current_finders = AtomicUsize::new(finders.len());
        run_finders(ctx, finders, solution_tx, pause, &current_finders, progress)
    }
}

/// Runs multiple finders in parallel via. Rayon.
fn run_finders<const CUBOIDS: usize>(
    ctx: &FinderCtx<CUBOIDS>,
    finders: &[FinderInfo<CUBOIDS>],
    solution_tx: &mpsc::Sender<Solution>,
    pause: &AtomicBool,
    current_finders: &AtomicUsize,
    progress: Option<&ProgressBar>,
) -> anyhow::Result<Vec<FinderInfo<CUBOIDS>>> {
    match finders {
        [] => Ok(Vec::new()),
        [finder] => run_finder(
            ctx,
            Finder::new(ctx, finder)?,
            solution_tx,
            pause,
            current_finders,
            progress,
        ),
        _ => {
            let mid = finders.len() / 2;
            let (res_a, res_b) = rayon::join(
                || {
                    run_finders(
                        ctx,
                        &finders[..mid],
                        solution_tx,
                        pause,
                        current_finders,
                        progress,
                    )
                },
                || {
                    run_finders(
                        ctx,
                        &finders[mid..],
                        solution_tx,
                        pause,
                        current_finders,
                        progress,
                    )
                },
            );

            let mut finders = res_a?;
            finders.append(&mut res_b?);
            Ok(finders)
        }
    }
}

/// Runs a `Finder` until it finishes or `pause` is set to true, sending its
/// results through `solution_tx` and splitting itself if `current_finders` gets
/// too low.
fn run_finder<const CUBOIDS: usize>(
    ctx: &FinderCtx<CUBOIDS>,
    mut finder: Finder<CUBOIDS>,
    solution_tx: &mpsc::Sender<Solution>,
    pause: &AtomicBool,
    current_finders: &AtomicUsize,
    progress: Option<&ProgressBar>,
) -> anyhow::Result<Vec<FinderInfo<CUBOIDS>>> {
    let mut send_counter: u16 = 0;
    loop {
        if finder.index < finder.queue.len() {
            // Evaluate the next instruction in the queue.
            finder.handle_instruction(ctx);
        } else {
            // We've reached the end of the queue. So, finalize the current net to find
            // solutions and send them off.
            for solution in finder.finalize(ctx) {
                solution_tx.send(solution)?;
            }

            // Now backtrack and look for more solutions.
            if !finder.backtrack() {
                // Backtracking failed which means there are no solutions left and we're done.
                current_finders.fetch_sub(1, Ordering::Relaxed);
                if let Some(progress) = progress {
                    progress.inc(1);
                }
                return Ok(Vec::new());
            }
        }

        send_counter = send_counter.wrapping_add(1);
        if send_counter == 0 {
            if pause.load(Ordering::Relaxed) {
                return Ok(vec![finder.into_info(ctx)]);
            } else if current_finders.load(Ordering::Relaxed) < rayon::current_num_threads() + 1 {
                // Split this `Finder` if there aren't currently enough of them to give all
                // our threads something to do.
                if let Some(new_finder) = finder.split(ctx) {
                    current_finders.fetch_add(1, Ordering::Relaxed);
                    if let Some(progress) = progress {
                        progress.inc_length(1);
                    }

                    let (res_a, res_b) = rayon::join(
                        || run_finder(ctx, finder, solution_tx, pause, current_finders, progress),
                        || {
                            run_finder(
                                ctx,
                                new_finder,
                                solution_tx,
                                pause,
                                current_finders,
                                progress,
                            )
                        },
                    );
                    let mut finders = res_a?;
                    finders.append(&mut res_b?);
                    return Ok(finders);
                }
            }
        }
    }
}
