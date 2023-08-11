//! A crate which finds nets for cubiods.

// This file contains infrastructure shared between `primary.rs` and `alt.rs`.

use std::{array, cmp::Reverse, iter::zip};

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

/// Generates a list of equivalence classes out of all possible cursors on the
/// given cuboids, except for the one specified by `fixed_cuboid`, which alweys
/// uses `fixed_cursor` instead.
fn equivalence_classes_inner<const CUBOIDS: usize>(
    cuboids: [Cuboid; CUBOIDS],
    square_caches: &[SquareCache; CUBOIDS],
    fixed_cuboid: usize,
    fixed_cursor: CursorData,
) -> Vec<SkipSet<CUBOIDS>> {
    let mut result: Vec<SkipSet<CUBOIDS>> = Vec::new();

    let mut cursor_choices: Vec<_> = cuboids
        .iter()
        .map(|cuboid| cuboid.unique_cursors())
        .collect();
    cursor_choices[fixed_cuboid] = vec![fixed_cursor];
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

/// Returns the average size of an equivalence class in a list of equivalence
/// classes.
fn avg_size<const CUBOIDS: usize>(classes: &[SkipSet<CUBOIDS>]) -> f64 {
    let total_size = classes
        .iter()
        .map(|class| class.canon_mappings().len())
        .sum::<usize>();
    total_size as f64 / classes.len() as f64
}

/// Returns a list of equivalence classes of mappings which all lead to the same
/// set of nets when used as starting positions.
///
/// This does not cover all possible mappings, just enough to make sure that we
/// get all the solutions.
pub fn equivalence_classes<const CUBOIDS: usize>(
    cuboids: [Cuboid; CUBOIDS],
    square_caches: &[SquareCache; CUBOIDS],
) -> Vec<SkipSet<CUBOIDS>> {
    // Go through all the possible mappings we could fix and find the one which
    // results in the equivalence classes having the largest average size, since
    // this leads to the most mappings getting skipped.
    let mut result = (0..cuboids.len())
        .flat_map(|cuboid| {
            cuboids[cuboid]
                .unique_cursors()
                .into_iter()
                .map(move |cursor| {
                    equivalence_classes_inner(cuboids, square_caches, cuboid, cursor)
                })
        })
        .max_by(|a, b| {
            avg_size(a)
                .partial_cmp(&avg_size(b))
                .unwrap()
                // If two lists of equivalence classes have the same average size pick the shorter
                // one.
                .then(b.len().cmp(&a.len()))
        })
        .unwrap();
    // Then sort the equivalence classes in descending order of size, so that more
    // mappings get skipped earlier on.
    result.sort_by_key(|class| Reverse(class.canon_mappings().len()));
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
///
/// A potential future optimisation would be to do an additional
/// canonicalisation step of rotating the cursors in tandem until the first one
/// is orientation 0. However this would make equivalence class optimisation
/// annoying because `canon_mappings().len()` would no longer correspond
/// directly to the number of actual cursors represented by the mapping.
///
/// It does so currently because the number of equivalent cursors for any cursor
/// is constant across a cuboid: if all the dimensions are different, each
/// cursor has 4 equivalents; if two of the dimensions are the same, they have
/// 8; and if all of them are the same, they have 24.
///
/// On the other hand, while discarding different orientations can collapse *up
/// to* 4 mappings into 1, it can be less if cursors are equivalent to rotations
/// of themselves (which I believe can only happen if they're in the middle of a
/// face).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkipSet<const CUBOIDS: usize> {
    /// For each cuboid, a lookup table containing the canon version of every
    /// cursor on that cuboid.
    #[serde(with = "crate::utils::arrays")]
    canon_lookups: [Vec<Cursor>; CUBOIDS],
    /// The set of mappings with canonical cursors.
    inner: FxHashSet<Mapping<CUBOIDS>>,
}

impl<const CUBOIDS: usize> SkipSet<CUBOIDS> {
    /// Creates a new `SkipSet` for mappings based on the given list of square
    /// caches.
    pub fn new(square_caches: &[SquareCache; CUBOIDS]) -> Self {
        let canon_lookups = array::from_fn(|i| {
            let cache = &square_caches[i];
            cache
                .squares()
                .flat_map(|square| (0..4).map(move |orientation| Cursor::new(square, orientation)))
                .map(|cursor| Cursor::from_data(cache, &cursor.to_data(cache).canon()))
                .collect()
        });
        Self {
            canon_lookups,
            inner: FxHashSet::default(),
        }
    }

    /// Returns whether `mapping` is contained within this set.
    pub fn contains(&self, mut mapping: Mapping<CUBOIDS>) -> bool {
        // Replace all the mapping's cursors with their canonical versions.
        for (lookup, cursor) in zip(&self.canon_lookups, mapping.cursors_mut()) {
            *cursor = lookup[usize::from(cursor.0)];
        }
        // Then check if our inner set contains that canonicalised mapping.
        self.inner.contains(&mapping)
    }

    /// Inserts a mapping and all of the mappings with equivalent cursors into
    /// this set.
    pub fn insert(&mut self, mut mapping: Mapping<CUBOIDS>) {
        // Replace all the mapping's cursors with their canonical versions.
        for (lookup, cursor) in zip(&self.canon_lookups, mapping.cursors_mut()) {
            *cursor = lookup[usize::from(cursor.0)];
        }
        // Then insert it into the inner set.
        self.inner.insert(mapping);
    }

    /// Returns an iterator over the inner list of canon mappings.
    pub fn canon_mappings(&self) -> impl Iterator<Item = &Mapping<CUBOIDS>> + ExactSizeIterator {
        self.inner.iter()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Cuboid, Mapping, SquareCache};

    #[test]
    fn equivalence_classes() {
        let cuboids = [Cuboid::new(1, 1, 7), Cuboid::new(1, 3, 3)];
        let square_caches = cuboids.map(SquareCache::new);
        let equivalence_classes = super::equivalence_classes(cuboids, &square_caches);
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
