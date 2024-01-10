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
use net_finder::{equivalence_classes, Class, Cuboid, Cursor, Direction, Square, SquareCache};

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

    let (fixed_cuboid, fixed_class, _) = equivalence_classes(&square_caches);
    let mapping_index_bits: usize = square_caches
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != fixed_cuboid)
        .map(|(_, cache)| cache.classes().len().next_power_of_two().ilog2() as usize)
        .sum();

    let mapping_index_module = gen_mapping_index(&square_caches, fixed_cuboid, fixed_class);
    let neighbour_offset_function = gen_neighbour_offset(&square_caches);

    let mut file = File::create(output)?;
    write!(
        file,
        "\
parameter int CUBOIDS = {CUBOIDS};
parameter int AREA = {area};

typedef logic [$clog2(AREA)-1:0] square_t;
typedef struct packed {{
  square_t square;
  logic [1:0] orientation;
}} cursor_t;
typedef cursor_t [CUBOIDS-1:0] mapping_t;
typedef logic[{mapping_index_bits}-1:0] mapping_index_t;

{mapping_index_module}

{neighbour_offset_function}
"
    )?;
    Ok(())
}

fn gen_mapping_index<const CUBOIDS: usize>(
    square_caches: &[SquareCache; CUBOIDS],
    fixed_cuboid: usize,
    fixed_class: Class,
) -> String {
    let class_bits = square_caches
        .iter()
        .map(|cache| cache.classes().len().next_power_of_two().ilog2() as usize)
        .collect::<Vec<_>>();

    let class_funcs = square_caches
        .iter()
        .zip(&class_bits)
        .enumerate()
        .map(|(cuboid, (cache, &bits))| {
            format!(
                "function automatic logic [{bits}-1:0] cuboid{cuboid}_class(cursor_t cursor);\
               \n  case (cursor)\
               \n    {}\
               \n    default: return 'x;\
               \n  endcase\
               \nendfunction",
                cache
                    .classes()
                    .map(|class| format!(
                        "{cursors}: return {index};",
                        index = class.index(),
                        cursors = class.contents(cache).map(|cursor| cursor.0).format(", "),
                    ))
                    .format("\n    "),
            )
        })
        .join("\n\n");

    let undo_funcs = square_caches
        .iter()
        .zip(&class_bits)
        .enumerate()
        .filter(|&(cuboid, _)| cuboid != fixed_cuboid)
        .map(|(cuboid, (cache, bits))| {
            format!(
                "function automatic logic [{top_bit}:0] cuboid{cuboid}_undo(\
               \n  logic [{top_bit}:0] klass,\
               \n  logic [2:0] transform\
               \n);\
               \n  logic [2:0] new_transform;\
               \n  case ({{klass, transform}})\
               \n    {}\
               \n    default: new_transform = 'x;\
               \n  endcase\
               \n\
               \n  return {{klass[{top_bit}:3], new_transform}};\
               \nendfunction",
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
                                    "{}: new_transform = {};",
                                    (class.index() as usize) << 3 | transform,
                                    // Include the full lower 3 bits of the transformed class,
                                    // rather than padding out the transform with 0s. This means we
                                    // can blindly replace the lower 3 bits of the class with this
                                    // without having to find out how many bits of it are actually
                                    // the new transform and how many are just the class.
                                    class.with_transform(new_transform).index() & 0b111,
                                )
                            })
                    })
                    .format("\n    "),
                top_bit = bits - 1,
            )
        })
        .join("\n");

    let class_signals = class_bits
        .iter()
        .enumerate()
        .map(|(cuboid, bits)| format!("logic [{bits}-1:0] cursor{cuboid}_class;"))
        .join("\n  ");
    let class_assignments = (0..CUBOIDS)
        .map(|cuboid| {
            format!("assign cursor{cuboid}_class = cuboid{cuboid}_class(mapping[{cuboid}]);")
        })
        .join("\n  ");

    // Figure out the list of transformations we have to try undoing given a
    // particular class on the fixed cuboid.

    // Every transformation that the fixed class doesn't care about doubles the
    // number of different transformations we have to consider.
    let num_options = 1 << (3 - fixed_class.transform_bits());

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
            format!(
                "case (cursor{fixed_cuboid}_class)\
           \n      {}\
           \n      default: to_undo[{i}] = 'x;\
           \n    endcase",
                map.iter()
                    .map(|(class, transform)| {
                        format!("{}: to_undo[{i}] = {transform};", class.index())
                    })
                    .format("\n      ")
            )
        })
        .join("\n    ");

    let uses_fixed_class_expr = to_undo[0]
        .keys()
        .map(|class| format!("cursor{fixed_cuboid}_class == {}", class.index()))
        .join(" | ");

    let undo_exprs = (0..CUBOIDS)
        .filter(|&cuboid| cuboid != fixed_cuboid)
        .map(|cuboid| format!("cuboid{cuboid}_undo(cursor{cuboid}_class, to_undo[i])"))
        .join(", ");

    format!(
        "{class_funcs}\
       \n\
       \n{undo_funcs}\
       \n\
       \nmodule mapping_index_lookup(\
       \n    input mapping_t mapping,\
       \n    output mapping_index_t index,\
       \n    output logic uses_fixed_class\
       \n);\
       \n  {class_signals}\
       \n  {class_assignments}\
       \n\
       \n  logic [2:0] to_undo[{num_options}];\
       \n  always_comb begin\
       \n    {to_undo_assignments}\
       \n  end\
       \n\
       \n  mapping_index_t options[{num_options}];\
       \n  always_comb begin\
       \n    // Make this as big as possible to start with so everything's less than it (or\
       \n    // equal, in which case it happens to be correct anyway and so that's fine).\
       \n    index = ~0;\
       \n    for (int i = 0; i < {num_options}; i++) begin\
       \n      options[i] = {{{undo_exprs}}};\
       \n      // Pick the smallest version of the index.\
       \n      if (options[i] < index) index = options[i];\
       \n    end\
       \n  end\
       \n\
       \n  assign uses_fixed_class = {uses_fixed_class_expr};\
       \nendmodule"
    )
}

fn gen_neighbour_offset<const CUBOIDS: usize>(square_caches: &[SquareCache; CUBOIDS]) -> String {
    // For each cuboid, a map from offset to every square + direction which gives a
    // neighbour with that offset from the original square (interpreted as a cursor
    // with orientation 0).
    let mut cursor_offsets: [BTreeMap<i32, Vec<(Square, Direction)>>; CUBOIDS] =
        array::from_fn(|_| BTreeMap::new());

    for (cache, offsets) in zip(square_caches, &mut cursor_offsets) {
        for square in cache.squares() {
            for direction in [Left, Up, Right, Down] {
                let cursor = Cursor::new(square, 0);
                let neighbour = cursor.moved_in(cache, direction);
                let offset = neighbour.0 as i32 - cursor.0 as i32;
                // double-check that this offset works for all orientations
                for orientation in 1..=3 {
                    let cursor = Cursor::new(square, orientation);
                    let neighbour = cursor.moved_in(cache, direction.turned(-orientation));
                    if neighbour.orientation() < orientation {
                        // Naively adding the offset here would lead cause the cursor to overflow
                        // into the next square, instead of just having the orientation overflow
                        // like it should; so the real offset is 4 less.
                        assert_eq!(neighbour.0 as i32 - cursor.0 as i32, offset - 4);
                    } else {
                        assert_eq!(neighbour.0 as i32 - cursor.0 as i32, offset);
                    }
                }
                offsets
                    .entry(offset)
                    .or_insert_with(Vec::new)
                    .push((square, direction))
            }
        }
    }

    format!(
        "function automatic int neighbour_offset(int cuboid, square_t square, logic [1:0] direction);\
       \n  {}\
       \n  else return 'x;\
       \nendfunction",
        cursor_offsets
            .iter()
            .enumerate()
            .map(|(cuboid, offsets)| format!(
                "if (cuboid == {cuboid})\
             \n    case ({{square, direction}})\
             \n      {}\
             \n      default: return 'x;\
             \n    endcase",
                offsets
                    .iter()
                    .map(|(offset, squares)| format!(
                        "{}: return {offset};",
                        squares
                            .iter()
                            .map(|&(square, direction)|
                                ((square.0 as u32) << 2) | direction as u32
                            )
                            .format(", ")
                    ))
                    .format("\n      ")
            ))
            .format("\n  else ")
    )
}
