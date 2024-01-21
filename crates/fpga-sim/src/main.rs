use std::ffi::CStr;
use std::num::NonZeroUsize;
use std::time::Duration;

use clap::Parser;
use net_finder::{Class, ClassMapping, Finder, FinderCtx, FinderInfo};
use net_finder_fpga_sim::{Core, Event, VerilatedContext, CUBOIDS};
use pretty_assertions::assert_eq;

#[derive(Parser)]
struct Args {
    steps: usize,
    split_step: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let core_ctx = VerilatedContext::new(true);
    let mut core = Core::new(
        &core_ctx,
        Some(CStr::from_bytes_with_nul(b"core.fst\0").unwrap()),
    );
    core.reset();

    let info = FinderInfo {
        start_mapping: ClassMapping::new([Class(138), Class(193), Class(200)]),
        decisions: vec![true],
        base_decision: NonZeroUsize::new(1).unwrap(),
    };

    let ctx = FinderCtx::new(CUBOIDS, Duration::ZERO)?;
    let mut finder = Finder::new(&ctx, &info)?;

    core.load_finder(&ctx, &info);

    for step in 0..args.steps {
        if step == args.split_step {
            *core.req_split() = true;
            core.update();
            while !matches!(core.event(&ctx), Event::Split(_)) {}
            *core.req_split() = false;
            core.update();

            finder.split(&ctx);
        }

        let (_, finder_success) = finder.step(&ctx);

        *core.out_ready() = true;
        let core_success = core.step();
        *core.out_ready() = false;

        if !finder_success {
            // Drop these before potentially panicking.
            drop(core);
            drop(core_ctx);

            println!("finder finished");
            assert!(!core_success);

            return Ok(());
        }
    }

    let core_info = core.pause(&ctx);
    let finder_info = finder.clone().into_info(&ctx);

    // Drop these before potentially panicking.
    drop(core);
    drop(core_ctx);

    println!("{finder:#?}");
    if let Some(core_info) = core_info {
        println!(
            "{}",
            core_info
                .decisions
                .iter()
                .map(|&decision| if decision { '1' } else { '0' })
                .collect::<String>()
        );
        println!(
            "{}",
            finder_info
                .decisions
                .iter()
                .map(|&decision| if decision { '1' } else { '0' })
                .collect::<String>()
        );

        assert_eq!(core_info, finder_info);
    } else {
        assert!(!finder.step(&ctx).1);
    }

    Ok(())
}
