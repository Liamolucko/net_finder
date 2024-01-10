#![no_main]
use std::env;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::bail;
use libfuzzer_sys::{fuzz_target, Corpus};
use net_finder::{Finder, FinderCtx, FinderInfo};
use net_finder_fpga_sim::{Core, VerilatedContext, CUBOIDS, NUM_CUBOIDS};
use pretty_assertions::assert_eq;

fuzz_target!(|input: (FinderInfo<NUM_CUBOIDS>, u8)| -> Corpus {
    static CTX: OnceLock<FinderCtx<NUM_CUBOIDS>> = OnceLock::new();
    let ctx = CTX.get_or_init(|| FinderCtx::new(CUBOIDS, Duration::ZERO).unwrap());
    let core_ctx = VerilatedContext::new(false);
    match main(ctx, &core_ctx, input.0, input.1) {
        Ok(()) => Corpus::Keep,
        Err(_) => Corpus::Reject,
    }
});

fn main(
    ctx: &FinderCtx<NUM_CUBOIDS>,
    core_ctx: &VerilatedContext,
    info: FinderInfo<NUM_CUBOIDS>,
    steps: u8,
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
    let mut core = Core::new(&core_ctx);
    core.reset();
    core.load_finder(&ctx, &info);

    // Run both of them for the same number of steps.
    let mut done = false;
    for _ in 0..steps {
        let finder_success = finder.step(&ctx);

        // Ignore any solutions the core outputs.
        *core.out_ready() = true;
        let core_success = core.step();
        *core.out_ready() = false;

        assert_eq!(core_success, finder_success);
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
        assert!(!finder.step(&ctx));
    }

    Ok(())
}
