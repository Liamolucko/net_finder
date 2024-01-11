#![no_main]
use std::env;
use std::num::NonZeroUsize;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::bail;
use libfuzzer_sys::{fuzz_target, Corpus};
use net_finder::{Finder, FinderCtx, FinderInfo};
use net_finder_fpga_sim::{Core, Event, VerilatedContext, CUBOIDS, NUM_CUBOIDS};
use pretty_assertions::assert_eq;

fuzz_target!(|input: (FinderInfo<NUM_CUBOIDS>, u8, u8)| -> Corpus {
    static CTX: OnceLock<FinderCtx<NUM_CUBOIDS>> = OnceLock::new();
    let ctx = CTX.get_or_init(|| FinderCtx::new(CUBOIDS, Duration::ZERO).unwrap());
    let core_ctx = VerilatedContext::new(false);
    match main(ctx, &core_ctx, input.0, input.1, input.2) {
        Ok(()) => Corpus::Keep,
        Err(_) => Corpus::Reject,
    }
});

fn main(
    ctx: &FinderCtx<NUM_CUBOIDS>,
    core_ctx: &VerilatedContext,
    info: FinderInfo<NUM_CUBOIDS>,
    steps: u8,
    split_step: u8,
) -> anyhow::Result<()> {
    if info.start_mapping.classes[ctx.fixed_cuboid].transform() != 0 {
        bail!(
            "fixed class is not root of family (`Finder::new` automatically corrects this but \
             then the output `FinderInfo` isn't the same causing fuzzing to fail)"
        );
    }

    // Create the `Finder` first, since it'll validate the `FinderInfo`.
    let mut finder = Finder::new(&ctx, &info)?;
    // Then create the `Core` and load the `FinderInfo` into it.
    let mut core = Core::new(&core_ctx, None);
    core.reset();
    core.load_finder(&ctx, &info);

    // Run both of them for the same number of steps.
    let mut done = false;
    for step in 0..steps {
        // If the `Core` produces a solution before splitting we need to save it and
        // check that the `Finder` produces the same solution a moment later.
        let mut cached_event = None;
        if step == split_step {
            // We split here and make sure the `Core` and `Finder` produce the
            // same result, and continue working the same afterwards.
            *core.req_split() = true;
            core.update();
            let core_result = loop {
                match core.event(ctx) {
                    // This is fine, it just means that the core is waiting until it has an
                    // instruction to split on. It's the equivalent of calling `handle_instruction`
                    // in `Finder::split` when we haven't tried running the instruction before a
                    // potential new `base_index` yet.
                    Event::Step => {}
                    // Save it for later.
                    event @ Event::Solution(_) => cached_event = Some(event),
                    Event::Split(info) => break Some(info),
                    Event::Pause(_) => panic!("unexpected pause"),
                    Event::Receiving => break None,
                }
            };
            *core.req_split() = false;
            core.update();

            let finder_result = finder
                .split(ctx)
                .map(|new_finder| new_finder.into_info(ctx));

            if env::var_os("PRINT_FINDER").is_some() {
                println!("{:#?}", finder);
            }

            assert_eq!(core_result, finder_result);
        }

        let (solution, finder_success) = finder.step(ctx);
        if let Some(mut finder_solution) = solution {
            let event = cached_event.unwrap_or_else(|| core.event(ctx));
            let Event::Solution(mut core_solution) = event else {
                panic!("expected Event::Solution but got {event:#?}")
            };

            // We don't care what the `base_decision`s of the solutions are, since they only
            // affect backtracking which we don't need to do when processing solutions.
            //
            // The reason they can differ is because in the event that we happen to split
            // right as a solution is found (i.e., right after the step that runs the last
            // instruction in the queue), the core will emit the solution before it notices
            // the request to split whereas the `Finder` will split first.
            //
            // This means that the core's solution will have the `base_decision` from before
            // splitting, and the `Finder`'s one will have the `base_decision` from after.
            core_solution.base_decision = NonZeroUsize::new(1).unwrap();
            finder_solution.base_decision = NonZeroUsize::new(1).unwrap();

            assert_eq!(core_solution, finder_solution)
        }

        if finder_success {
            assert_eq!(core.event(ctx), Event::Step)
        } else {
            assert_eq!(core.event(ctx), Event::Receiving)
        }

        if !finder_success {
            done = true;
        }
    }

    if env::var_os("PRINT_FINDER").is_some() {
        println!("{:#?}", finder);
    }
    if done {
        assert_eq!(core.pause(&ctx), None);
    } else if let Some(info) = core.pause(&ctx) {
        assert_eq!(info, finder.into_info(&ctx));
    } else {
        assert!(!finder.step(&ctx).1);
    }

    Ok(())
}
