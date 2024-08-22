//! Code for generating the contents of the FPGA implementation's various RAMs.

use std::iter;

use crate::{Class, Cursor, Direction, FinderCtx, SquareCache};
use Direction::*;

/// Returns the contents of all the cuboids' neighbour lookups.
pub fn neighbour_lookups<const CUBOIDS: usize>(ctx: &FinderCtx<CUBOIDS>) -> Vec<Vec<u64>> {
    ctx.square_caches
        .iter()
        .enumerate()
        .map(|(i, cache)| neighbour_lookup(cache, i == 0, ctx.fixed_class))
        .collect()
}

/// Returns the contents of a specific cuboid's neighbour lookup.
fn neighbour_lookup(cache: &SquareCache, fixed_cuboid: bool, fixed_class: Class) -> Vec<u64> {
    let t_mode = [Left, Up, Right, Down].into_iter().flat_map(|direction| {
        cache
            .squares()
            .map(move |square| {
                let middle = Cursor::new(square, 0).moved_in(cache, direction);

                // Rotate the directions around so that the ones we need to generate are in
                // indices 1, 2 and 3.
                let mut directions = [Left, Up, Right, Down];
                directions.rotate_left(direction.turned(2) as usize);

                let neighbours =
                    directions.map(|direction| middle.moved_in(cache, direction).0 as u64);

                let mut result = middle.0 as u64
                    | (neighbours[1] << 8)
                    | (neighbours[2] << 16)
                    | (neighbours[3] << 24);

                if fixed_cuboid && middle.class(cache).root() == fixed_class {
                    result |= 1 << 32;
                    result |= (middle.class(cache).transform() as u64) << 33;
                }

                result
            })
            .chain(iter::repeat(0))
            .take(64)
    });

    let normal_mode = cache
        .squares()
        .map(|square| {
            let middle = Cursor::new(square, 0);

            let neighbours =
                [Left, Up, Right, Down].map(|direction| middle.moved_in(cache, direction).0 as u64);

            let mut result = neighbours[0]
                | (neighbours[1] << 8)
                | (neighbours[2] << 16)
                | (neighbours[3] << 24);

            if fixed_cuboid && middle.class(cache).root() == fixed_class {
                result |= 1 << 32;
                result |= (middle.class(cache).transform() as u64) << 33;
            }

            result
        })
        .chain(iter::repeat(0))
        .take(64);

    t_mode.chain(normal_mode).collect()
}

/// Returns the contents of all the cuboids' neighbour lookups.
pub fn undo_lookups<const CUBOIDS: usize>(ctx: &FinderCtx<CUBOIDS>) -> Vec<Vec<u16>> {
    ctx.square_caches
        .iter()
        .skip(1)
        .map(|cache| undo_lookup(&ctx.square_caches[0], cache, ctx.fixed_class))
        .collect()
}

/// Returns the contents of a specific cuboid's neighbour lookup.
fn undo_lookup(
    fixed_square_cache: &SquareCache,
    cache: &SquareCache,
    fixed_class: Class,
) -> Vec<u16> {
    assert!(fixed_class.transform_bits() >= 2, "The FPGA's skip checker assumes the fixed class has at least 2 transform bits (which in true in practice until at least area 64)");
    cache
        .cursors()
        .flat_map(|cursor| {
            (0..8).map(move |transform| {
                let class = cursor.class(cache);
                let options = fixed_class
                    .with_transform(transform)
                    .alternate_transforms(fixed_square_cache)
                    .map(|transform| class.undo_transform(cache, transform))
                    .collect::<Vec<_>>();

                assert!(options.len() <= 2);

                let hi = (options[0].index() >> 3) as u16;
                let lo0 = (options[0].index() & 0x07) as u16;
                let lo1 = (options.get(1).unwrap_or(&options[0]).index() & 0x07) as u16;

                lo0 | (lo1 << 3) | (hi << 6)
            })
        })
        .collect()
}
