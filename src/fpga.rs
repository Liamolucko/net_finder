//! Code for generating the contents of the FPGA implementation's various RAMs.

use std::iter;

use crate::{Class, Cursor, Direction, FinderCtx, SquareCache};
use Direction::*;

pub fn clog2(x: usize) -> u32 {
    usize::BITS - (x - 1).leading_zeros()
}

pub fn max_run_stack_len(max_area: usize) -> usize {
    max_area - 1
}

pub fn max_decisions_len(max_area: usize) -> usize {
    1 + 4 + 3 * (max_run_stack_len(max_area) - 1)
}

/// Returns the contents of all the cuboids' neighbour lookups.
pub fn neighbour_lookups<const CUBOIDS: usize>(
    ctx: &FinderCtx<CUBOIDS>,
    max_area: usize,
    cuboids: usize,
) -> Vec<Vec<u64>> {
    ctx.square_caches
        .iter()
        .enumerate()
        .chain(iter::repeat((0, &ctx.square_caches[0])))
        .take(cuboids)
        .map(|(i, cache)| neighbour_lookup(max_area, cache, i == 0, ctx.fixed_class))
        .collect()
}

/// Returns the contents of a specific cuboid's neighbour lookup.
fn neighbour_lookup(
    max_area: usize,
    cache: &SquareCache,
    fixed_cuboid: bool,
    fixed_class: Class,
) -> Vec<u64> {
    fn pack_entry(
        max_area: usize,
        neighbours: [Cursor; 4],
        fixed_family: bool,
        transform: u8,
    ) -> u64 {
        let cursor_bits = clog2(4 * max_area);

        let mut result = neighbours[0].0 as u64
            | ((neighbours[1].0 as u64) << cursor_bits)
            | ((neighbours[2].0 as u64) << (2 * cursor_bits))
            | ((neighbours[3].0 as u64) << (3 * cursor_bits));

        if fixed_family {
            result |= 1 << (4 * cursor_bits);
            result |= (transform as u64) << (4 * cursor_bits + 1);
        }

        result
    }

    let t_mode = [Left, Up, Right, Down].into_iter().flat_map(|direction| {
        cache
            .squares()
            .map(move |square| {
                let middle = Cursor::new(square, 0).moved_in(cache, direction);

                // Rotate the directions around so that the ones we need to generate are in
                // indices 1, 2 and 3.
                let mut directions = [Left, Up, Right, Down];
                directions.rotate_left(direction.turned(2) as usize);

                let neighbours = directions.map(|direction| middle.moved_in(cache, direction));

                pack_entry(
                    max_area,
                    [middle, neighbours[1], neighbours[2], neighbours[3]],
                    fixed_cuboid && middle.class(cache).root() == fixed_class,
                    middle.class(cache).transform(),
                )
            })
            .chain(iter::repeat(0))
            .take(max_area)
    });

    let normal_mode = cache
        .squares()
        .map(|square| {
            let middle = Cursor::new(square, 0);

            let neighbours =
                [Left, Up, Right, Down].map(|direction| middle.moved_in(cache, direction));

            pack_entry(
                max_area,
                neighbours,
                fixed_cuboid && middle.class(cache).root() == fixed_class,
                middle.class(cache).transform(),
            )
        })
        .chain(iter::repeat(0))
        .take(max_area);

    t_mode.chain(normal_mode).collect()
}

/// Returns the contents of all the cuboids' neighbour lookups.
pub fn undo_lookups<const CUBOIDS: usize>(
    ctx: &FinderCtx<CUBOIDS>,
    cuboids: usize,
) -> Vec<Vec<u16>> {
    ctx.square_caches
        .iter()
        .skip(1)
        .chain(iter::repeat(&ctx.square_caches[0]))
        .take(cuboids - 1)
        .map(|cache| undo_lookup(&ctx.square_caches[0], cache, ctx.fixed_class))
        .collect()
}

/// Returns the contents of a specific cuboid's undo lookup.
fn undo_lookup(
    fixed_square_cache: &SquareCache,
    cache: &SquareCache,
    fixed_class: Class,
) -> Vec<u16> {
    assert!(
        fixed_class.transform_bits() >= 2,
        "The FPGA's skip checker assumes the fixed class has at least 2 transform bits (which is \
         true in practice until at least area 64)"
    );
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

pub fn fmt_bits(bits: impl IntoIterator<Item = bool>) -> String {
    bits.into_iter()
        .map(|bit| if bit { '1' } else { '0' })
        .collect()
}

pub fn parse_bits(bits: &str) -> Vec<bool> {
    bits.chars()
        .map(|c| match c {
            '0' => false,
            '1' => true,
            _ => unreachable!(),
        })
        .collect()
}
