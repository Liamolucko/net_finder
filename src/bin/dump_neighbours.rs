//! Dumps the information required by `test_neighbour_lookup.py` as JSON to stdout.

use std::io;
use std::time::Duration;

use net_finder::{fpga, Combinations, Cuboid, Cursor, Direction, FinderCtx, Mapping, Pos};
use serde_json::json;

use Direction::*;

fn main() {
    let ctx = FinderCtx::new([Cuboid::new(1, 1, 5), Cuboid::new(1, 2, 3)], Duration::ZERO).unwrap();

    let neighbour_lookup_contents = fpga::neighbour_lookups(&ctx);

    let cursor_choices = ctx.square_caches.each_ref().map(|cache| {
        cache
            .squares()
            .flat_map(|square| (0..4).map(move |orientation| Cursor::new(square, orientation)))
            .collect::<Vec<_>>()
    });
    let instructions = Combinations::new(&cursor_choices)
        .enumerate()
        .map(|(i, cursors)| {
            let x = i % 64;
            let y = i / 64;
            (
                // The actual values don't matter, we're just trying to test all the
                // possibilities, so it's fine if this wraps around.
                Pos::new(x as u8, y as u8),
                Mapping::<2>::new(cursors.try_into().unwrap()),
            )
        })
        .collect::<Vec<_>>();

    fn serialize_instruction(pos: Pos, mapping: Mapping<2>) -> serde_json::Value {
        json!({
            "pos": {
                "x": pos.x % 64,
                "y": pos.y % 64,
            },
            "mapping": mapping.cursors.map(|cursor| cursor.0)
        })
    }

    let t_mode = instructions.iter().flat_map(|&(pos, mapping)| {
        [Left, Up, Right, Down].map(|direction| {
            let middle_pos = pos.moved_in_unchecked(direction);
            let middle_mapping = mapping.moved_in(&ctx.square_caches, direction);

            let neighbours = [Left, Up, Right, Down].map(|direction| {
                (
                    middle_pos.moved_in_unchecked(direction),
                    middle_mapping.moved_in(&ctx.square_caches, direction),
                )
            });

            json!({
                "input": serialize_instruction(pos, mapping),
                "t_mode": true,
                "direction": direction as u8,
                "middle": serialize_instruction(middle_pos, middle_mapping),
                "neighbours": neighbours.map(|(pos, mapping)| serialize_instruction(pos, mapping)),
                "fixed_family": ctx.fixed_family(middle_mapping.cursors[0]),
                "transform": middle_mapping.cursors[0].class(&ctx.square_caches[0]).transform()
            })
        })
    });

    let normal_mode = instructions.iter().enumerate().map(|(i, &(pos, mapping))| {
        let neighbours = [Left, Up, Right, Down].map(|direction| {
            (
                pos.moved_in_unchecked(direction),
                mapping.moved_in(&ctx.square_caches, direction),
            )
        });

        json!({
            "input": serialize_instruction(pos, mapping),
            "t_mode": false,
            // Make sure this doesn't have an effect when `t_mode` is false.
            "direction": i % 4,
            "middle": serialize_instruction(pos, mapping),
            "neighbours": neighbours.map(|(pos, mapping)| serialize_instruction(pos, mapping)),
            "fixed_family": ctx.fixed_family(mapping.cursors[0]),
            "transform": mapping.cursors[0].class(&ctx.square_caches[0]).transform()
        })
    });

    serde_json::to_writer(
        io::stdout(),
        &json!({
            "neighbour_lookup_contents": neighbour_lookup_contents,
            "test_cases": t_mode.chain(normal_mode).collect::<Vec<_>>(),
        }),
    )
    .unwrap();
}
