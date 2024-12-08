#![no_main]
use std::env;
use std::num::NonZero;
use std::num::NonZeroUsize;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::bail;
use async_executor::LocalExecutor;
use futures_concurrency::prelude::*;
use libfuzzer_sys::{fuzz_target, Corpus};
use net_finder::{Cuboid, Finder, FinderCtx, FinderInfo};
use net_finder_fpga_sim::{Core, Event, VerilatedContext};
use pretty_assertions::assert_eq;

const CUBOIDS2: [[Cuboid; 2]; 4] = [
    [Cuboid::new(1, 1, 5), Cuboid::new(1, 2, 3)],
    [Cuboid::new(1, 1, 7), Cuboid::new(1, 3, 3)],
    [Cuboid::new(1, 1, 8), Cuboid::new(1, 2, 5)],
    [Cuboid::new(1, 1, 9), Cuboid::new(1, 3, 4)],
];

const CUBOIDS3: [[Cuboid; 3]; 5] = [
    [
        Cuboid::new(1, 1, 11),
        Cuboid::new(1, 2, 7),
        Cuboid::new(1, 3, 5),
    ],
    [
        Cuboid::new(1, 1, 13),
        Cuboid::new(1, 3, 6),
        Cuboid::new(3, 3, 3),
    ],
    [
        Cuboid::new(1, 1, 14),
        Cuboid::new(1, 2, 9),
        Cuboid::new(1, 4, 5),
    ],
    [
        Cuboid::new(1, 1, 15),
        Cuboid::new(1, 3, 7),
        Cuboid::new(2, 3, 5),
    ],
    [
        Cuboid::new(1, 2, 10),
        Cuboid::new(2, 2, 7),
        Cuboid::new(2, 4, 4),
    ],
];

static CONTEXTS2: LazyLock<[FinderCtx<2>; 4]> =
    LazyLock::new(|| CUBOIDS2.map(|cuboids| FinderCtx::new(cuboids, Duration::ZERO).unwrap()));
static CONTEXTS3: LazyLock<[FinderCtx<3>; 5]> =
    LazyLock::new(|| CUBOIDS3.map(|cuboids| FinderCtx::new(cuboids, Duration::ZERO).unwrap()));

fuzz_target!(
    |input: (bool, u8, [(u16, Vec<bool>, NonZero<u8>, u8, u8); 4])| -> Corpus {
        let (three, ctx_index, input) = input;
        let result = if three {
            let ctx = &CONTEXTS3[ctx_index as usize % CONTEXTS3.len()];
            main(ctx, input)
        } else {
            let ctx = &CONTEXTS2[ctx_index as usize % CONTEXTS2.len()];
            main(ctx, input)
        };

        match result {
            Ok(_) => Corpus::Keep,
            Err(_) => Corpus::Reject,
        }
    }
);

fn main<const CUBOIDS: usize>(
    ctx: &FinderCtx<CUBOIDS>,
    input: [(u16, Vec<bool>, NonZero<u8>, u8, u8); 4],
) -> anyhow::Result<()> {
    let core_ctx = VerilatedContext::new(false);
    let core = Core::new(&core_ctx, None);
    core.reset();
    core.fill_mems(&ctx);

    let fut = input
        .into_iter()
        .enumerate()
        .map(
            |(i, (equivalence_class, mut decisions, base_decision, steps, split_step))| {
                decisions.insert(0, true);
                drive_interface(
                    ctx,
                    &core,
                    i,
                    equivalence_class.try_into().unwrap(),
                    decisions,
                    base_decision.try_into().unwrap(),
                    steps,
                    split_step,
                )
            },
        )
        .collect::<Vec<_>>()
        .try_join();

    let executor = LocalExecutor::new();
    let task = executor.spawn(fut);
    while !executor.is_empty() {
        while executor.try_tick() {}
        core.clock();
    }

    pollster::block_on(executor.run(task)).map(|vec| vec.into_iter().collect())
}

async fn drive_interface<const CUBOIDS: usize>(
    ctx: &FinderCtx<CUBOIDS>,
    core: &Core<'_>,
    interface_num: usize,
    equivalence_class: usize,
    decisions: Vec<bool>,
    base_decision: NonZero<usize>,
    steps: u8,
    split_step: u8,
) -> anyhow::Result<()> {
    let interface = core.interfaces().nth(interface_num).unwrap();

    let mut start_mappings = ctx.start_mappings();
    let start_mapping = start_mappings
        .nth(equivalence_class % start_mappings.len())
        .unwrap();
    let info = FinderInfo {
        start_mapping,
        base_decision: NonZero::new((base_decision.get() - 1) % decisions.len() + 1).unwrap(),
        decisions,
    };

    // The base decision should always be a 1; if it's a 0, it may as well be moved
    // to the next decision since it can't be backtracked again anyway.
    if info.decisions.get(info.base_decision.get()) == Some(&false) {
        bail!("`base_decision` does not point to a 1");
    }

    // Create the `Finder` first, since it'll validate the `FinderInfo`.
    let mut finder = Finder::new(&ctx, &info)?;
    // Then load the `FinderInfo` into our `Interface`.
    interface.load_finder(&ctx, &info).await;

    // Run both of them for the same number of steps.
    let mut done = false;
    for step in 0..steps {
        // If the `Core` produces a solution before splitting we need to save it and
        // check that the `Finder` produces the same solution a moment later.
        let mut cached_event = None;
        if step == split_step {
            // We split here and make sure the `Core` and `Finder` produce the
            // same result, and continue working the same afterwards.
            interface.req_split.set(true);
            interface.core().update();

            // Wait for it to switch into Split state so that we can turn `req_split` off
            // again and avoid accidentally triggering a second split right after the first.
            while !interface.out_valid.get() && !interface.wants_finder.get() {
                interface.core().tick().await;
            }
            interface.req_split.set(false);
            interface.core().update();

            let core_result = loop {
                match interface.event(ctx).await {
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

            let finder_result = finder
                .split(ctx)
                .map(|new_finder| new_finder.into_info(ctx));

            if env::var_os("PRINT_FINDER").is_some() {
                println!("{:#?}", finder);
            }

            assert_eq!(core_result, finder_result, "interface {interface_num}");
        }

        let (solution, finder_success) = finder.step(ctx);
        if let Some(mut finder_solution) = solution {
            let event = match cached_event {
                Some(event) => event,
                None => interface.event(ctx).await,
            };
            let Event::Solution(mut core_solution) = event else {
                panic!("interface {interface_num}: expected Event::Solution but got {event:#?}")
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

            assert_eq!(core_solution, finder_solution, "interface {interface_num}")
        }

        if finder_success {
            assert_eq!(
                interface.event(ctx).await,
                Event::Step,
                "interface {interface_num}"
            )
        } else {
            assert_eq!(
                interface.event(ctx).await,
                Event::Receiving,
                "interface {interface_num}"
            )
        }

        if !finder_success {
            done = true;
        }
    }

    if env::var_os("PRINT_FINDER").is_some() {
        println!("{:#?}", finder);
    }
    if done {
        assert_eq!(
            interface.pause(&ctx).await,
            None,
            "interface {interface_num}"
        );
    } else if let Some(info) = interface.pause(&ctx).await {
        assert_eq!(info, finder.into_info(&ctx), "interface {interface_num}");
    } else {
        assert!(!finder.step(&ctx).1, "interface {interface_num}");
    }

    Ok(())
}
