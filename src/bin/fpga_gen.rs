use std::array;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write as _;
use std::iter::zip;
use std::path::PathBuf;

use anyhow::bail;
use clap::Parser;
use itertools::Itertools;
use net_finder::{equivalence_classes, Class, Cuboid, Cursor, Direction, SquareCache};

use Direction::*;

#[derive(Parser)]
struct Options {
    output: PathBuf,
    cuboids: Vec<Cuboid>,
}

fn main() -> anyhow::Result<()> {
    let Options { output, cuboids } = Options::parse();
    match cuboids.as_slice() {
        &[] => run(output, []),
        &[a] => run(output, [a]),
        &[a, b] => run(output, [a, b]),
        &[a, b, c] => run(output, [a, b, c]),
        _ => bail!("only up to 3 cuboids supported"),
    }
}

fn run<const CUBOIDS: usize>(output: PathBuf, cuboids: [Cuboid; CUBOIDS]) -> anyhow::Result<()> {
    for cuboid in cuboids.into_iter().skip(1) {
        assert_eq!(cuboid.surface_area(), cuboids[0].surface_area());
    }
    let area = cuboids[0].surface_area();

    let square_caches = cuboids.map(SquareCache::new);

    let cursor_bits = (4 * area).next_power_of_two().ilog2() as usize;
    let mapping_index_bits: usize = square_caches
        .iter()
        .map(|cache| cache.classes().len().next_power_of_two().ilog2() as usize)
        .sum();

    // Round the size of the net up to the nearest multiple of 4 to make our net
    // layout work.
    let net_size = area.next_multiple_of(4);
    let coord_bits = net_size.next_power_of_two().ilog2();
    // Multiplication logic is expensive, so we index into the net by just gluing
    // the bits of x and y together; this means that we have to store an entry for
    // every combination of y's bits (`1 << coord_bits`), but x is still normal
    // since it's at the start.
    let net_len = net_size * (1 << coord_bits);

    let (skip_checker_entity, skip_checker_component) = gen_skip_checker(area, &square_caches);
    let (neighbour_lookup_entity, neighbour_lookup_component) =
        gen_neighbour_lookup(area, &square_caches);

    let mut file = File::create(output)?;
    write!(
        file,
        "\
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package generated is
\tconstant cuboids: integer := {CUBOIDS};
\tconstant area: integer := {area};
\tconstant net_size: integer := {net_size};
\tconstant net_len: integer := {net_len};

\tconstant cursor_bits: integer := {cursor_bits};
\tconstant coord_bits: integer := {coord_bits};

\tsubtype cursor is unsigned(cursor_bits - 1 downto 0);
\ttype mapping is array(0 to cuboids - 1) of cursor;
\tsubtype mapping_index is unsigned({mapping_index_bits} - 1 downto 0);

\tsubtype square is integer range 0 to area - 1;
\ttype square_vector is array(integer range <>) of square;

\ttype pos is record
\t\tx: unsigned(coord_bits - 1 downto 0);
\t\ty: unsigned(coord_bits - 1 downto 0);
\tend record pos;
\ttype pos_vector is array(integer range <>) of pos;

\tsubtype direction is unsigned(1 downto 0);
\ttype direction_vector is array(integer range <>) of direction;

\ttype instruction is record
\t\tpos: pos;
\t\tmapping: mapping;
\tend record instruction;
\ttype instruction_vector is array(integer range <>) of instruction;

\t{skip_checker_component}

\t{neighbour_lookup_component}
end package;

{skip_checker_entity}

{neighbour_lookup_entity}
"
    )?;
    Ok(())
}

fn gen_skip_checker<const CUBOIDS: usize>(
    area: usize,
    square_caches: &[SquareCache; CUBOIDS],
) -> (String, String) {
    let (fixed_cuboid, fixed_class, _) = equivalence_classes(&square_caches);

    let cursor_bits = (4 * area).next_power_of_two().ilog2() as usize;
    let class_bits = square_caches
        .iter()
        .map(|cache| cache.classes().len().next_power_of_two().ilog2() as usize)
        .collect::<Vec<_>>();

    let undo_entities = square_caches
        .iter()
        .zip(&class_bits)
        .enumerate()
        .map(|(cuboid, (cache, bits))| {
            format!(
                "library ieee;\n\
                 use ieee.std_logic_1164.all;\n\
                 \n\
                 entity cuboid{cuboid}_undo is\n\
                 \tport(\n\
                 \t\tclass: in std_logic_vector({top_bit} downto 0);\n\
                 \t\ttransform: in std_logic_vector(2 downto 0);\n\
                 \t\ttransformed: out std_logic_vector({top_bit} downto 0));\n\
                 end cuboid{cuboid}_undo;\n\
                 \n\
                 architecture arch of cuboid{cuboid}_undo is\n\
                 \tsignal new_transform: std_logic_vector(2 downto 0);\n\
                 begin\n\
                 \twith class & transform select\n\
                 \t\tnew_transform <=\n\
                 \t\t\t{},\n\
                 \t\t\t\"---\" when others;\n\
                 \ttransformed <= class({top_bit} downto 3) & new_transform;\n\
                 end arch;",
                cache
                    .classes()
                    .flat_map(|class| {
                        class
                            .contents(cache)
                            .next()
                            .unwrap()
                            .undo_lookup(cache)
                            .into_iter()
                            .enumerate()
                            .map(move |(transform, new_transform)| {
                                format!(
                                    "\"{:03b}\" when \"{:0bits$b}{:03b}\"",
                                    new_transform,
                                    class.index(),
                                    transform,
                                )
                            })
                    })
                    .format(",\n\t\t\t"),
                top_bit = bits - 1,
            )
        })
        .join("\n");

    let undo_components = class_bits
        .iter()
        .enumerate()
        .map(|(cuboid, bits)| {
            format!(
                "component cuboid{cuboid}_undo is\n\
                 \t\tport(\n\
                 \t\t\tclass: in std_logic_vector({top_bit} downto 0);\n\
                 \t\t\ttransform: in std_logic_vector(2 downto 0);\n\
                 \t\t\ttransformed: out std_logic_vector({top_bit} downto 0));\n\
                 \tend component;",
                top_bit = bits - 1
            )
        })
        .join("\n\t");

    // Every transformation that the fixed class doesn't care about doubles the
    // number of different transformations we have to consider.
    let num_options = 1 << (3 - fixed_class.transform_bits());

    let class_signals = class_bits
        .iter()
        .enumerate()
        .map(|(cuboid, bits)| {
            format!(
                "signal cursor{cuboid}_class: std_logic_vector({top_bit} downto 0);\n\
                 \ttype cuboid{cuboid}_class_vector is array(integer range <>) of std_logic_vector({top_bit} downto 0);\n\
                 \tsignal class{cuboid}_transformed: cuboid{cuboid}_class_vector({num_options} - 1 downto 0);",
                top_bit = bits - 1
            )
        })
        .join("\n\t");

    let class_assignments = square_caches
        .iter()
        .zip(&class_bits)
        .enumerate()
        .map(|(cuboid, (cache, bits))| {
            format!(
                "with mapping({cuboid}) select\n\
                 \t\tcursor{cuboid}_class <=\n\
                 \t\t\t{},\n\
                 \t\t\t\"{}\" when others;",
                cache
                    .classes()
                    .map(|class| format!(
                        "\"{index:0bits$b}\" when {cursors}",
                        index = class.index(),
                        cursors = class
                            .contents(cache)
                            .map(|cursor| format!("\"{:0cursor_bits$b}\"", cursor.0))
                            .format(" | "),
                    ))
                    .format(",\n\t\t\t"),
                "-".repeat(*bits)
            )
        })
        .join("\n\t");

    // Figure out the list of transformations we have to try undoing given a
    // particular class on the fixed cuboid.

    // For each index in `to_undo` (the VHDL signal), what that index should be set
    // to for each possible class on the fixed cuboid.
    let mut to_undo: Vec<HashMap<Class, u8>> = vec![HashMap::new(); num_options];
    for transform in 0..8 {
        let cache = &square_caches[fixed_cuboid];
        let mut cursor = fixed_class.contents(cache).next().unwrap().to_data(cache);
        if transform & 0b100 != 0 {
            cursor.horizontal_flip();
        }
        cursor.orientation = (cursor.orientation + (transform & 0b11) as i8) & 0b11;
        let transformed = cache
            .classes()
            .find(|class| {
                class
                    .contents(cache)
                    .contains(&Cursor::from_data(cache, &cursor))
            })
            .unwrap();
        for map in to_undo.iter_mut() {
            match map.entry(transformed) {
                Entry::Occupied(_) => {}
                Entry::Vacant(entry) => {
                    entry.insert(transform);
                    break;
                }
            }
        }
    }

    for class in to_undo[0].keys() {
        for map in to_undo.iter() {
            assert!(map.contains_key(class))
        }
    }

    let to_undo_assignments = to_undo
        .iter()
        .enumerate()
        .map(|(i, map)| {
            let bits = class_bits[fixed_cuboid];
            format!(
                "with cursor{fixed_cuboid}_class select\n\
                 \t\tto_undo({i}) <=\n\
                 \t\t\t{},\n\
                 \t\t\t\"---\" when others;",
                map.iter()
                    .map(|(class, transform)| {
                        format!("\"{transform:03b}\" when \"{:0bits$b}\"", class.index())
                    })
                    .format(",\n\t\t\t")
            )
        })
        .join("\n\t");

    let uses_fixed_class_assignment = format!(
        "uses_fixed_class <= '1' when {} else '0';",
        to_undo[0]
            .keys()
            .map(|class| {
                let bits = class_bits[fixed_cuboid];
                format!(
                    "(cursor{fixed_cuboid}_class = \"{:0bits$b}\")",
                    class.index()
                )
            })
            .format(" or ")
    );

    let undo_instances = (0..CUBOIDS).filter(|&cuboid| cuboid != fixed_cuboid)
        .flat_map(|cuboid| {
            (0..num_options)
                .map(move |i| format!("class{cuboid}_option{i}: cuboid{cuboid}_undo port map(cursor{cuboid}_class, to_undo({i}), class{cuboid}_transformed({i}));"))
        })
        .join("\n\t");

    let result_assignment = format!(
        "skipped <= '1' when uses_fixed_class = '1' and ({}) else '0';",
        (0..num_options)
            .map(|i| {
                format!(
                    "{} < start_mapping_index",
                    (0..CUBOIDS)
                        .filter(|&cuboid| cuboid != fixed_cuboid)
                        .map(|cuboid| format!("unsigned(class{cuboid}_transformed({i}))"))
                        .format(" & ")
                )
            })
            .format(" or ")
    );

    (
        format!(
            "\
{undo_entities}

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.generated.all;

entity skip_checker is
\tport(
\t\tmapping: in mapping;
\t\tstart_mapping_index: in mapping_index;
\t\tskipped: out std_logic);
end skip_checker;

architecture arch of skip_checker is
\t{undo_components}
\t{class_signals}
\tsignal uses_fixed_class: std_logic;
\ttype transform_vector is array(integer range <>) of std_logic_vector(2 downto 0);
\tsignal to_undo: transform_vector({num_options} - 1 downto 0);
begin
\t{class_assignments}
\t{to_undo_assignments}
\t{uses_fixed_class_assignment}
\t{undo_instances}
\t{result_assignment}
end arch;"
        ),
        format!(
            "\
component skip_checker is
\t\tport(
\t\t\tmapping: in mapping;
\t\t\tstart_mapping_index: in mapping_index;
\t\t\tskipped: out std_logic);
\tend component;"
        ),
    )
}

fn gen_neighbour_lookup<const CUBOIDS: usize>(
    area: usize,
    square_caches: &[SquareCache; CUBOIDS],
) -> (String, String) {
    let square_bits = area.next_power_of_two().ilog2() as usize;
    let cursor_bits = square_bits + 2;

    // For each cuboid, a map from offset to every cursor + direction which gives a
    // neighbour with that offset from the original cursor.
    let mut cursor_offsets: [BTreeMap<i32, Vec<(Cursor, Direction)>>; CUBOIDS] =
        array::from_fn(|_| BTreeMap::new());

    for (cache, offsets) in zip(square_caches, &mut cursor_offsets) {
        for cursor in cache
            .squares()
            .flat_map(|square| (0..4).map(move |orientation| Cursor::new(square, orientation)))
        {
            for direction in [Left, Up, Right, Down] {
                let neighbour = cursor.moved_in(cache, direction);
                let offset = neighbour.0 as i32 - cursor.0 as i32;
                offsets
                    .entry(offset)
                    .or_insert_with(Vec::new)
                    .push((cursor, direction))
            }
        }
    }

    let offset_signals = cursor_offsets
        .iter()
        .enumerate()
        .map(|(i, offsets)| {
            let min_offset = offsets.keys().min().unwrap();
            let max_offset = offsets.keys().max().unwrap();
            format!("signal offset{i}: integer range {min_offset} to {max_offset};")
        })
        .join("\n\t");

    let cursor_assignments = cursor_offsets
        .iter()
        .enumerate()
        .map(|(i, offsets)| {
            format!(
                "with unsigned'(instruction.mapping({i}) & direction) select\n\
                 \t\toffset{i} <=\n\
                 \t\t\t{},\n\
                 \t\t\t0 when others;\n\
                 \n\
                 \tneighbour.mapping({i}) <= to_unsigned(to_integer(instruction.mapping({i})) + offset{i}, cursor_bits);",
                offsets
                    .iter()
                    .map(|(offset, cursors)| {
                        format!(
                            "{offset} when {}",
                            cursors
                                .iter()
                                .map(|&(cursor, direction)| format!(
                                    "\"{:0cursor_bits$b}{:02b}\"",
                                    cursor.0, direction as u8
                                ))
                                .format(" | ")
                        )
                    })
                    .format(",\n\t\t\t"),
            )
        })
        .join("\n\n\t");

    (
        format!(
            "\
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.generated.all;

entity neighbour_lookup is
\tport(
\t\tinstruction: in instruction;
\t\tdirection: in direction;
\t\tneighbour: out instruction);
end neighbour_lookup;

architecture arch of neighbour_lookup is
\t{offset_signals}
begin
\tprocess(instruction)
\tbegin
\t\tneighbour.pos <= instruction.pos;
\t\tcase direction is
\t\t\twhen \"00\" =>
\t\t\t\tif to_integer(instruction.pos.x) = 0 then
\t\t\t\t\tneighbour.pos.x <= to_unsigned(net_size - 1, coord_bits);
\t\t\t\telse
\t\t\t\t\tneighbour.pos.x <= instruction.pos.x - 1;
\t\t\t\tend if;
\t\t\twhen \"01\" =>
\t\t\t\tif to_integer(instruction.pos.y) = net_size - 1 then
\t\t\t\t\tneighbour.pos.y <= (others => '0');
\t\t\t\telse
\t\t\t\t\tneighbour.pos.y <= instruction.pos.y + 1;
\t\t\t\tend if;
\t\t\twhen \"10\" =>
\t\t\t\tif to_integer(instruction.pos.x) = net_size - 1 then
\t\t\t\t\tneighbour.pos.x <= (others => '0');
\t\t\t\telse
\t\t\t\t\tneighbour.pos.x <= instruction.pos.x + 1;
\t\t\t\tend if;
\t\t\twhen \"11\" =>
\t\t\t\tif to_integer(instruction.pos.y) = 0 then
\t\t\t\t\tneighbour.pos.y <= to_unsigned(net_size - 1, coord_bits);
\t\t\t\telse
\t\t\t\t\tneighbour.pos.y <= instruction.pos.y - 1;
\t\t\t\tend if;
\t\tend case;
\tend process;

\t{cursor_assignments}
end arch;"
        ),
        format!(
            "\
component neighbour_lookup is
\t\tport(
\t\t\tinstruction: in instruction;
\t\t\tdirection: in direction;
\t\t\tneighbour: out instruction);
\tend component;"
        ),
    )
}
