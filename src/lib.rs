//! A crate which finds nets for cubiods.

// This file contains infrastructure shared between `primary.rs` and `alt.rs`.

use std::iter::zip;

mod geometry;
mod primary;
mod utils;
mod zdd;

pub use geometry::*;
pub use primary::*;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};
use utils::Combinations;
pub use zdd::*;

/// Returns a list of equivalence classes of mappings which all lead to the same
/// set of nets when used as starting positions.
///
/// All mappings between these cuboids can be categorised into one of these
/// classes.
pub fn equivalence_classes(cuboids: &[Cuboid], square_caches: &[SquareCache]) -> Vec<SkipSet> {
    let mut result: Vec<SkipSet> = Vec::new();

    let cursor_choices: Vec<_> = cuboids
        .iter()
        .map(|cuboid| cuboid.unique_cursors())
        .collect();
    for combination in Combinations::new(&cursor_choices) {
        let mapping_data = MappingData::new(combination);
        let mapping = Mapping::from_data(square_caches, &mapping_data);
        if !result.iter().any(|class| class.contains(mapping)) {
            // We've found a mapping that's in a new equivalence class. Add it and all its
            // equivalents to the list.
            let mut set = SkipSet::new(square_caches);
            // `SkipSet::insert` already implicitly inserts all the mappings with equivalent
            // cursors, but that's not quite enough: we still need to manually insert the
            // mappings you get by rotating both cursors in tandem, and flipping the
            // mapping.
            for rotation in 0..4 {
                let mut mapping_data = mapping_data.clone();
                // Add the rotated version of the mapping.
                for cursor in &mut mapping_data.cursors {
                    cursor.orientation = (cursor.orientation + rotation) & 0b11;
                }
                set.insert(Mapping::from_data(square_caches, &mapping_data));
                // Then all the combinations of vertical and horizontal flips.
                mapping_data.horizontal_flip();
                set.insert(Mapping::from_data(square_caches, &mapping_data));
                mapping_data.vertical_flip();
                set.insert(Mapping::from_data(square_caches, &mapping_data));
                mapping_data.horizontal_flip();
                set.insert(Mapping::from_data(square_caches, &mapping_data));
            }
            result.push(set)
        }
    }

    result
}

/// A set of mappings for a `NetFinder` to skip.
///
/// This is better than storing a set of mappings directly because it only
/// stores one mapping to represent every mapping whose cursors are equivalent
/// to that mapping's cursors, saving a lot of space.
///
/// Internally, when checking if the set contains a mapping, this first looks up
/// the canon versions of the mapping's cursors and then checks if the mapping
/// with those cursors is in the set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkipSet {
    /// For each cuboid, a lookup table containing the canon version of every
    /// cursor on that cuboid.
    canon_lookups: Vec<Vec<Cursor>>,
    /// The set of mappings with canonical cursors.
    inner: FxHashSet<Mapping>,
}

impl SkipSet {
    /// Creates a new `SkipSet` for mappings based on the given list of square
    /// caches.
    pub fn new(square_caches: &[SquareCache]) -> Self {
        Self {
            canon_lookups: square_caches
                .iter()
                .map(|cache| {
                    cache
                        .squares()
                        .flat_map(|square| {
                            (0..4).map(move |orientation| Cursor::new(square, orientation))
                        })
                        .map(|cursor| Cursor::from_data(cache, &cursor.to_data(cache).canon()))
                        .collect()
                })
                .collect(),
            inner: FxHashSet::default(),
        }
    }

    /// Returns whether `mapping` is contained within this set.
    pub fn contains(&self, mut mapping: Mapping) -> bool {
        // Replace all the mapping's cursors with their canonical versions.
        for (lookup, cursor) in zip(&self.canon_lookups, mapping.cursors_mut()) {
            *cursor = lookup[usize::from(cursor.0)];
        }
        // Then check if our inner set contains that canonicalised mapping.
        self.inner.contains(&mapping)
    }

    /// Inserts a mapping and all of the mappings with equivalent cursors into
    /// this set.
    pub fn insert(&mut self, mut mapping: Mapping) {
        // Replace all the mapping's cursors with their canonical versions.
        for (lookup, cursor) in zip(&self.canon_lookups, mapping.cursors_mut()) {
            *cursor = lookup[usize::from(cursor.0)];
        }
        // Then insert it into the inner set.
        self.inner.insert(mapping);
    }

    /// Returns an iterator over the inner list of canon mappings.
    pub fn canon_mappings(&self) -> impl Iterator<Item = &Mapping> {
        self.inner.iter()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Cuboid, Mapping, SquareCache};

    #[test]
    fn equivalence_classes() {
        let cuboids = [Cuboid::new(1, 1, 7), Cuboid::new(1, 3, 3)];
        let square_caches: Vec<_> = cuboids
            .iter()
            .map(|&cuboid| SquareCache::new(cuboid))
            .collect();
        let equivalence_classes = super::equivalence_classes(&cuboids, &square_caches);
        for class in equivalence_classes {
            for mapping in class.canon_mappings() {
                let mapping_data = mapping.to_data(&square_caches);
                for equivalent_data in mapping_data.equivalents() {
                    let equivalent = Mapping::from_data(&square_caches, &equivalent_data);
                    assert!(class.contains(equivalent));
                }
            }
        }
    }
}
