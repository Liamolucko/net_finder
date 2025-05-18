use std::time::Duration;

use divan::Bencher;
use net_finder::{Cuboid, Finder, FinderCtx};
use net_finder_gpu::Pipeline;

fn main() {
    divan::main();
}

#[divan::bench(args = [
    vec![Cuboid::new(1, 1, 5), Cuboid::new(1, 2, 3)],
    vec![Cuboid::new(1, 1, 7), Cuboid::new(1, 3, 3)],
    vec![Cuboid::new(1, 1, 8), Cuboid::new(1, 2, 5)],
    vec![Cuboid::new(1, 1, 9), Cuboid::new(1, 3, 4)],
    vec![Cuboid::new(1, 1, 11), Cuboid::new(1, 2, 7), Cuboid::new(1, 3, 5)],
])]
fn bench(bencher: Bencher, cuboids: &Vec<Cuboid>) {
    match *cuboids.as_slice() {
        [a] => bench_inner(bencher, [a]),
        [a, b] => bench_inner(bencher, [a, b]),
        [a, b, c] => bench_inner(bencher, [a, b, c]),
        _ => panic!("only up to 3 cuboids are supported"),
    }
}

fn bench_inner<const CUBOIDS: usize>(bencher: Bencher, cuboids: [Cuboid; CUBOIDS]) {
    let ctx = FinderCtx::new(cuboids, Duration::ZERO).unwrap();
    let mut pipeline = pollster::block_on(Pipeline::new(ctx.clone(), 1000)).unwrap();
    let (_, finders, _, _) = pipeline.run_finders(
        ctx.gen_finders()
            .into_iter()
            .map(|info| Finder::new(&ctx, &info).unwrap())
            .collect(),
    );

    bencher.bench_local(|| pipeline.run_finders(finders.clone()));
}
