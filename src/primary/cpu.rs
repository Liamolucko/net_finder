//! The initial home-grown algorithm I came up with.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::BufWriter;
use std::iter::zip;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use indicatif::ProgressBar;

use crate::{Finder, Net, Solution, State};

use super::{state_path, FinderCtx};

/// Updates the passed `state` with the most recently sent `Finder`s, then
/// writes it out to a file.
fn update_state<const CUBOIDS: usize>(
    state: &mut State<CUBOIDS>,
    finder_receivers: &mut Vec<Receiver<Finder<CUBOIDS>>>,
    channel_rx: &mut Receiver<Receiver<Finder<CUBOIDS>>>,
    progress: &ProgressBar,
) {
    state.prior_search_time = progress.elapsed();

    // Check if there are any new `Finder`s we need to add to our list.
    loop {
        match channel_rx.try_recv() {
            Err(TryRecvError::Empty) => break,
            Ok(rx) => {
                // There should always be an initial state for the `Finder` sent
                // immediately after creating the channel.
                let finder = rx.try_recv().unwrap();
                state.finders.push(finder);
                finder_receivers.push(rx);
                progress.inc_length(1);
            }
            Err(TryRecvError::Disconnected) => {
                // If this was disconnected, the iterator must have been dropped. In that case
                // break from the loop, since we want to write the final state with all the
                // yielded nets.
                break;
            }
        }
    }

    let mut i = 0;
    while i < state.finders.len() {
        match finder_receivers[i].try_recv() {
            Ok(finder) => {
                state.finders[i] = finder;
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                // That `Finder` has finished; remove it from our lists.
                state.finders.remove(i);
                finder_receivers.remove(i);
                progress.inc(1);
                // Skip over incrementing `i`, since `i` corresponds to the next
                // `Finder` in the list now that we're removed one.
                continue;
            }
        }
        i += 1;
    }

    progress.tick();
}

fn write_state<const CUBOIDS: usize>(state: &State<CUBOIDS>) {
    let path = state_path(&state.cuboids);
    // Initially write to a temporary file so that the previous version is still
    // there if we get Ctrl+C'd while writing or something like that.
    let tmp_path = path.with_extension("json.tmp");
    let file = File::create(&tmp_path).unwrap();
    serde_json::to_writer(BufWriter::new(file), &state).unwrap();
    // Then move it to the real path.
    fs::rename(tmp_path, path).unwrap();
}

/// Runs a `Finder` to completion, sending its results and state updates
/// through the provided channels and splitting itself if `current_finders` gets
/// too low.
fn run_finder<'scope, const CUBOIDS: usize>(
    ctx: &'scope FinderCtx<CUBOIDS>,
    mut finder: Finder<CUBOIDS>,
    net_tx: Sender<Solution>,
    finder_tx: SyncSender<Finder<CUBOIDS>>,
    channel_tx: Sender<Receiver<Finder<CUBOIDS>>>,
    scope: &rayon::Scope<'scope>,
    current_finders: &'scope AtomicUsize,
) {
    let mut send_counter: u16 = 0;
    loop {
        if finder.index < finder.queue.len() {
            // Evaluate the next instruction in the queue.
            finder.handle_instruction(ctx);
        } else {
            // We've reached the end of the queue. So, finalize the current net to find
            // solutions and send them off.
            for solution in finder.finalize(ctx) {
                net_tx.send(solution).unwrap();
            }

            // Now backtrack and look for more solutions.
            if !finder.backtrack() {
                // Backtracking failed which means there are no solutions left and we're done.
                current_finders.fetch_sub(1, Relaxed);
                return;
            }
        }

        send_counter = send_counter.wrapping_add(1);
        if send_counter == 0 {
            let _ = finder_tx.try_send(finder.clone());
            if current_finders.load(Relaxed) < rayon::current_num_threads() + 1 {
                // Split this `Finder` if there aren't currently enough of them to give all
                // our threads something to do.
                if let Some(finder) = finder.split(ctx) {
                    current_finders.fetch_add(1, Relaxed);

                    // Make a `finder_tx` for the new `Finder` to use.
                    let (finder_tx, finder_rx) = mpsc::sync_channel(1);
                    finder_tx.send(finder.clone()).unwrap();
                    channel_tx.send(finder_rx).unwrap();

                    let net_tx = net_tx.clone();
                    let channel_tx = channel_tx.clone();
                    scope.spawn(move |s| {
                        run_finder(
                            ctx,
                            finder,
                            net_tx,
                            finder_tx,
                            channel_tx,
                            s,
                            current_finders,
                        )
                    });
                }
            }
        }
    }
}

pub fn run<const CUBOIDS: usize>(
    state: Arc<Mutex<State<CUBOIDS>>>,
    mut yielded_nets: HashSet<Net>,
    progress: ProgressBar,
) -> anyhow::Result<impl Iterator<Item = Solution>> {
    let guard = state.lock().unwrap();
    let State {
        cuboids,
        ref finders,
        prior_search_time,
        ..
    } = *guard;
    let ctx = FinderCtx::new(cuboids, prior_search_time)?;

    // Create a channel for sending yielded nets to the main thread.
    let (net_tx, net_rx) = mpsc::channel::<Solution>();
    // Create a channel for each `Finder` to periodically send its state through.
    let (finder_senders, mut finder_receivers) = finders
        .iter()
        .map(|_| mpsc::sync_channel(1))
        .unzip::<_, _, Vec<_>, Vec<_>>();
    // Then create a channel through which we send the receiving ends of new such
    // channels.
    let (channel_tx, mut channel_rx) = mpsc::channel();

    let current_finders = Arc::new(AtomicUsize::new(finders.len()));
    thread::Builder::new()
        .name("scope thread".to_owned())
        .spawn({
            let finders = finders.clone();
            move || {
                rayon::scope(|s| {
                    // Force these to get moved into the closure without moving `current_finders`
                    // into it as well (since then it gets dropped before the scope ends).
                    let net_tx = net_tx;
                    let channel_tx = channel_tx;
                    for (finder, finder_tx) in zip(finders, finder_senders) {
                        let net_tx = net_tx.clone();
                        let channel_tx = channel_tx.clone();
                        s.spawn(|s| {
                            run_finder(
                                &ctx,
                                finder,
                                net_tx,
                                finder_tx,
                                channel_tx,
                                s,
                                &current_finders,
                            )
                        })
                    }
                });
            }
        })
        .unwrap();

    drop(guard);

    Ok(std::iter::from_fn(move || loop {
        update_state(
            &mut state.lock().unwrap(),
            &mut finder_receivers,
            &mut channel_rx,
            &progress,
        );
        // Stop waiting every 50ms to update `state` in case we get Ctrl+C'd.
        match net_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(solution) => {
                let new = yielded_nets.insert(solution.net.clone());
                if new {
                    state.lock().unwrap().solutions.push(solution.clone());
                    return Some(solution);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                // Write out our final state, since it serves as our way of retrieving results
                // afterwards.
                write_state(&state.lock().unwrap());
                return None;
            }
        }
    }))
}
