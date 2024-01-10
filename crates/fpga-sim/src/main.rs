use std::num::NonZeroUsize;
use std::time::Duration;

use clap::Parser;
use net_finder::{Class, ClassMapping, Finder, FinderCtx, FinderInfo};
use net_finder_fpga_sim::{Core, VerilatedContext, CUBOIDS};
use pretty_assertions::assert_eq;

#[derive(Parser)]
struct Args {
    steps: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let core_ctx = VerilatedContext::new(true);
    let mut core = Core::new(&core_ctx);
    core.reset();

    let info = FinderInfo {
        start_mapping: ClassMapping::new([Class(143), Class(201), Class(200)]),
        decisions: vec![true, false, false, true, false, false, false, false],
        base_decision: NonZeroUsize::new(8).unwrap(),
    };

    let ctx = FinderCtx::new(CUBOIDS, Duration::ZERO)?;
    let mut finder = Finder::new(&ctx, &info)?;

    core.load_finder(&ctx, &info);
    for _ in 0..args.steps {
        let finder_success = finder.step(&ctx);

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
        assert!(!finder.step(&ctx));
    }

    Ok(())
}
