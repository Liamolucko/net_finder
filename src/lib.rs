//! A crate which finds nets for cubiods.

// This file contains infrastructure shared between `primary.rs` and `alt.rs`.

use std::collections::HashSet;

mod geometry;
mod primary;
mod utils;
mod zdd;

pub use geometry::*;
pub use primary::*;
use utils::Combinations;
pub use zdd::*;

/// Returns a list of equivalence classes of mappings which all lead to the same
/// set of nets when used as starting positions.
///
/// Accepts the index, `fix`, of the cuboid to fix to one particular starting
/// mapping; the regular `equivalence_classes` then tries all of them.
pub fn equivalence_classes_inner(cuboids: &[Cuboid], fix: usize) -> Vec<HashSet<MappingData>> {
    let mut result: Vec<HashSet<MappingData>> = Vec::new();

    let mut cursor_choices: Vec<_> = cuboids
        .iter()
        .map(|cuboid| cuboid.unique_cursors())
        .collect();
    cursor_choices[fix] = vec![CursorData::new(cuboids[fix])];
    for combination in Combinations::new(&cursor_choices) {
        let mapping = MappingData::new(combination);
        if !result.iter().any(|class| class.contains(&mapping)) {
            // We've found a mapping that's in a new equivalence class. Add it to the list.
            result.push(mapping.equivalents())
        }
    }

    result
}

/// Returns a list of equivalence classes of mappings which all lead to the same
/// set of nets when used as starting positions.
///
/// These classes do not cover all mappings, just enough of them to cover enough
/// starting points to result in all possible nets.
pub fn equivalence_classes(cuboids: &[Cuboid]) -> Vec<HashSet<MappingData>> {
    (0..cuboids.len())
        .map(|fix| equivalence_classes_inner(cuboids, fix))
        .min_by_key(|classes| classes.len())
        .unwrap_or_else(Vec::new)
}
