//! Dumps the information required by `test_skip_checker.py` as JSON to stdout.

use std::io;
use std::time::Duration;

use clap::Parser;
use net_finder::{fpga, Combinations, Cuboid, FinderCtx, Mapping};
use serde_json::json;

#[derive(Parser)]
struct Options {
    cuboids: Vec<Cuboid>,
}

fn main() {
    let Options { cuboids } = Options::parse();
    match *cuboids.as_slice() {
        [a] => run([a]),
        [a, b] => run([a, b]),
        [a, b, c] => run([a, b, c]),
        _ => panic!("only up to 3 cuboids supported"),
    }
}

fn run<const CUBOIDS: usize>(cuboids: [Cuboid; CUBOIDS]) {
    let ctx = FinderCtx::new(cuboids, Duration::ZERO).unwrap();

    let undo_lookup_contents = fpga::undo_lookups(&ctx, 2);

    let cursor_choices = ctx
        .square_caches
        .each_ref()
        .map(|cache| cache.cursors().collect::<Vec<_>>());
    let test_cases = ctx.gen_finders().into_iter().flat_map(|finder| {
        let ctx = &ctx;
        Combinations::new(&cursor_choices).map(move |cursors| {
            let start_mapping_index = ctx.to_inner(finder.start_mapping).index();
            json!({
                "start_mapping_index": start_mapping_index & ((1 << (6 * (CUBOIDS - 1))) - 1),
                "input": cursors,
                "fixed_family": ctx.fixed_family(cursors[0]),
                "transform": cursors[0].class(&ctx.square_caches[0]).transform(),
                "skip": ctx.skip(start_mapping_index, Mapping::new(cursors.try_into().unwrap()))
            })
        })
    });

    serde_json::to_writer(
        io::stdout(),
        &json!({
            "undo_lookup_contents": undo_lookup_contents,
            "test_cases": test_cases.collect::<Vec<_>>(),
        }),
    )
    .unwrap();
}
