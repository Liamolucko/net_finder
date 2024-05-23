#![cfg_attr(feature = "no-trie", feature(portable_simd))]

//! A crate which finds nets for cubiods.

// This file contains infrastructure shared between `primary.rs` and `alt.rs`.

use std::iter;
use std::ops::RangeFrom;

use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};

mod geometry;
mod primary;
mod utils;
mod zdd;

pub use geometry::*;
pub use primary::*;
pub use utils::*;
pub use zdd::*;

/// Generates a list of equivalence classes out of all possible cursors on the
/// given cuboids, except for the one specified by `fixed_cuboid`, which alweys
/// uses `fixed_cursor` instead.
fn equivalence_classes_inner<const CUBOIDS: usize>(
    square_caches: &[SquareCache; CUBOIDS],
    fixed_cuboid: usize,
    fixed_class: Class,
) -> Vec<FxHashSet<ClassMapping<CUBOIDS>>> {
    let mut result: Vec<FxHashSet<ClassMapping<CUBOIDS>>> = Vec::new();

    let mut class_choices: Vec<Vec<Class>> = square_caches
        .iter()
        .map(|cache| cache.classes().collect())
        .collect();
    class_choices[fixed_cuboid] = vec![fixed_class];
    for combination in Combinations::new(&class_choices) {
        let combination: ClassMapping<CUBOIDS> = ClassMapping::new(combination.try_into().unwrap());
        if !result.iter().any(|class| class.contains(&combination)) {
            // We've found a mapping that's in a new equivalence class. Add it and all its
            // equivalents to the list.
            //
            // We can only really calculate rotations/flips on concrete cursors, not
            // equivalence classes, so take the first cursor from each class and use that to
            // perform calculations.
            let mapping_data = combination.sample(square_caches).to_data(square_caches);

            let mut class = FxHashSet::default();
            // Building our mappings out of `Class`es means that they automatically include
            // all mappings with equivalent cursors, but that's not quite enough: we still
            // need to manually insert the mappings you get by rotating both cursors in
            // tandem, and flipping the mapping.
            for rotation in 0..4 {
                let mut mapping_data = mapping_data.clone();
                // Add the rotated version of the mapping.
                for cursor in &mut mapping_data.cursors {
                    cursor.orientation = (cursor.orientation + rotation) & 0b11;
                }
                class.insert(ClassMapping::from_data(square_caches, &mapping_data));
                // Then all the combinations of vertical and horizontal flips.
                mapping_data.horizontal_flip();
                class.insert(ClassMapping::from_data(square_caches, &mapping_data));
                mapping_data.vertical_flip();
                class.insert(ClassMapping::from_data(square_caches, &mapping_data));
                mapping_data.horizontal_flip();
                class.insert(ClassMapping::from_data(square_caches, &mapping_data));
            }
            result.push(class)
        }
    }

    result
}

/// Returns the average size of an equivalence class in a list of equivalence
/// classes.
fn avg_size<const CUBOIDS: usize>(classes: &[FxHashSet<ClassMapping<CUBOIDS>>]) -> f64 {
    let total_size = classes.iter().map(|class| class.len()).sum::<usize>();
    total_size as f64 / classes.len() as f64
}

/// Returns a list of equivalence classes of mappings which all lead to the same
/// set of nets when used as starting positions.
///
/// This does not cover all possible mappings, just enough to make sure that we
/// get all the solutions.
///
/// It also returns the cuboid on which all mappings have cursors within the
/// same class, as well as what that class is.
pub fn equivalence_classes<const CUBOIDS: usize>(
    square_caches: &[SquareCache; CUBOIDS],
) -> (usize, Class, Vec<FxHashSet<ClassMapping<CUBOIDS>>>) {
    // Go through all the possible cursor classes we could fix and find the one
    // which results in the equivalence classes having the largest average size,
    // since this leads to the most mappings getting skipped.
    let (fixed_cuboid, fixed_class, mut result) = (0..CUBOIDS)
        .flat_map(|cuboid| {
            square_caches[cuboid]
                .classes()
                // Always pick fixed classes which are the root of their family.
                // This shouldn't have any effect on the result, since you'll still get the same
                // sizes of equivalence classes no matter what the transform is, each mapping is
                // just transformed a bit.
                // TODO: if we ever get to area 64, we'll have to add a class.transform_bits() == 3
                // check here as well for the FPGA to be able to handle it.
                .filter(|class| class.transform() == 0)
                .map(move |class| {
                    (
                        cuboid,
                        class,
                        equivalence_classes_inner(square_caches, cuboid, class),
                    )
                })
        })
        .max_by(|(_, _, a), (_, _, b)| {
            avg_size(a)
                .partial_cmp(&avg_size(b))
                .unwrap()
                // If two lists of equivalence classes have the same average size pick the
                // shorter one.
                .then(b.len().cmp(&a.len()))
        })
        .unwrap();

    // Then sort the equivalence classes in ascending order of minimum mapping
    // index, since `Finder::skip` relies on that being the case (when `no-trie` is
    // enabled).
    result.sort_by_key(|class| {
        class
            .iter()
            .filter(|mapping| mapping.classes[fixed_cuboid] == fixed_class)
            .map(|mapping| mapping.index())
            .min()
    });

    (fixed_cuboid, fixed_class, result)
}

/// A set of mappings for a `Finder` to skip, implemented as a trie.
///
/// When inserting a mapping, all of the mappings with equivalent cursors are
/// automatically added as well. This is partially a holdover from a previous
/// implementation and partly to make it so that the trie can be pretty well
/// compressed (by mapping all equivalent cursors to the same page) without
/// having to put in too much effort.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkipSet<const CUBOIDS: usize> {
    /// This is implemented as a weird custom trie. It consists of a bunch of
    /// _pages_ which have one entry for every possible cursor on one of the
    /// cuboids; each entry is the index of the page in which to look up the
    /// next cursor in the mapping.
    ///
    /// This is with the exception of the last cursor in the mapping, where each
    /// entry is 1 bit for whether or not this set contains the given mapping.
    /// This means that the page consists of `ceil(page_size / 32)` array
    /// entries rather than the usual `page_size` entries.
    ///
    /// The page starting at index 0 is special, because it's where all mappings
    /// that are known to not be in the set prior to the last cursor get sent.
    /// All the entries are 0 so that anything in page 0 gets sent back to page
    /// 0, and then when looking at the last cursor it serves a dual purpose as
    /// a bitfield with all 0 bits.
    ///
    /// So, the page where you start searching starts at `page_size`.
    ///
    /// Note that nothing about this representation actually requires
    /// `SkipSet`'s 'insert all equivalents at once' behaviour; it just makes it
    /// slightly easier to construct by having all equivalent cursors map to the
    /// same page, since otherwise we'd need to do some kind of manual
    /// compression pass afterwards to avoid it being huge.
    data: Vec<u32>,
    /// The size of a page in the trie, aka the number of cursors on the cuboids
    /// this set is for, aka 4 * the surface area of those cuboids.
    ///
    /// Note that we only deal with mappings where all the cuboids are the same
    /// size.
    page_size: u32,
}

impl<const CUBOIDS: usize> SkipSet<CUBOIDS> {
    /// Creates a new `SkipSet` for mappings between the given cuboids.
    pub fn new(cuboids: [Cuboid; CUBOIDS]) -> Self {
        for cuboid in &cuboids[1..] {
            assert_eq!(cuboid.surface_area(), cuboids[0].surface_area());
        }
        let page_size = 4 * cuboids[0].surface_area();
        Self {
            page_size: page_size.try_into().unwrap(),
            // We start off with just page 0 and the starting page, with all of the starting page's
            // entries pointing to page 0.
            data: vec![0; 2 * page_size],
        }
    }

    /// Returns whether `mapping` is contained within this set.
    pub fn contains(&self, mapping: Mapping<CUBOIDS>) -> bool {
        // The index of the start of the page we're currently looking at.
        let mut page = self.page_size;
        // Go through all the normal levels of the trie.
        for cursor in mapping.cursors().iter().take(mapping.cursors().len() - 1) {
            page = self.data[usize::try_from(page).unwrap() + usize::from(cursor.0)];
        }
        // Then look up whether the trie actually contains `mapping` in the final
        // bitfield level of the trie.
        let index = usize::from(mapping.cursors().last().unwrap().0);
        (self.data[usize::try_from(page).unwrap() + index / 32] >> (index % 32)) & 1 != 0
    }

    /// Inserts a mapping and all of the mappings with equivalent cursors into
    /// this set.
    pub fn insert(&mut self, square_caches: &[SquareCache; CUBOIDS], mapping: Mapping<CUBOIDS>) {
        // The index of the start of the page we're currently looking at.
        let mut page = self.page_size;
        // Descend through the normal levels of the trie, adding new pages if any don't
        // exist yet.
        for (i, cursor) in mapping
            .cursors()
            .iter()
            .take(mapping.cursors().len() - 1)
            .enumerate()
        {
            let mut next_page = self.data[usize::try_from(page).unwrap() + usize::from(cursor.0)];
            if next_page == 0 {
                // There's no page for this prefix of mappings yet; make one.
                next_page = self.data.len().try_into().unwrap();
                self.data.extend(
                    iter::repeat(0).take(
                        usize::try_from(if i + 1 == CUBOIDS - 1 {
                            (self.page_size + 31) / 32
                        } else {
                            self.page_size
                        })
                        .unwrap(),
                    ),
                );
                // Update the entries for this cursor and all its equivalents in the current
                // page to point to the new page.
                for cursor in cursor.to_data(&square_caches[i]).equivalents() {
                    let cursor = Cursor::from_data(&square_caches[i], &cursor);
                    self.data[usize::try_from(page).unwrap() + usize::from(cursor.0)] = next_page;
                }
            }
            page = next_page;
        }
        // Then set the bits for the last cursor and all its equivalents in the last
        // page.
        for cursor in mapping
            .cursors()
            .last()
            .unwrap()
            .to_data(square_caches.last().unwrap())
            .equivalents()
        {
            let index = usize::from(Cursor::from_data(square_caches.last().unwrap(), &cursor).0);
            self.data[usize::try_from(page).unwrap() + index / 32] |= 1 << (index % 32);
        }
    }

    /// Returns the first included cursor whose value is within the given range
    /// on the given page.
    fn first_cursor(&self, level: usize, page: u32, range: RangeFrom<u8>) -> Option<Cursor> {
        let start = u32::from(range.start);
        if start >= self.page_size {
            return None;
        }
        if level < CUBOIDS - 1 {
            self.data[usize::try_from(page + start).unwrap()
                ..usize::try_from(page + self.page_size).unwrap()]
                .iter()
                .position(|&next_page| next_page != 0)
                .map(|index| u8::try_from(start + u32::try_from(index).unwrap()).unwrap())
        } else {
            // The first word in the page needs to be treated specially so we can exclude
            // bits prior to the start.
            let first_word =
                self.data[usize::try_from(page + start / 32).unwrap()] & !((1 << (start % 32)) - 1);
            if first_word != 0 {
                let prev_words = start / 32;
                Some(u8::try_from(prev_words * 32 + first_word.trailing_zeros()).unwrap())
            } else {
                let prev_words = start / 32 + 1;
                let word_index = self.data[usize::try_from(page + start / 32 + 1).unwrap()
                    ..=usize::try_from(page + (self.page_size - 1) / 32).unwrap()]
                    .iter()
                    .position(|&word| word != 0);
                word_index.map(|index| {
                    let word = self.data[usize::try_from(page + start / 32 + 1).unwrap() + index];
                    u8::try_from(
                        prev_words * 32
                            + u32::try_from(index * 32).unwrap()
                            + word.trailing_zeros(),
                    )
                    .unwrap()
                })
            }
        }
        .map(Cursor)
    }
}

impl<'a, const CUBOIDS: usize> IntoIterator for &'a SkipSet<CUBOIDS> {
    type Item = Mapping<CUBOIDS>;
    type IntoIter = SkipSetIter<'a, CUBOIDS>;

    fn into_iter(self) -> Self::IntoIter {
        SkipSetIter {
            set: self,
            prev_path: None,
        }
    }
}

pub struct SkipSetIter<'a, const CUBOIDS: usize> {
    set: &'a SkipSet<CUBOIDS>,
    // The path through the trie's pages we took last time, as well as the cursors we yielded.
    prev_path: Option<[(u32, Cursor); CUBOIDS]>,
}

impl<const CUBOIDS: usize> Iterator for SkipSetIter<'_, CUBOIDS> {
    type Item = Mapping<CUBOIDS>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut path = [(self.set.page_size, Cursor(0)); CUBOIDS];
        let filled = if let Some(prev_path) = self.prev_path {
            // Initialise `path` with our path from last time.
            path = prev_path;
            // Find the last stop along our path last time whose page still has another
            // valid cursor left on it.
            let (first_new_level, first_new_cursor) = prev_path
                .into_iter()
                .enumerate()
                .rev()
                .find_map(|(level, (page, cursor))| {
                    self.set
                        .first_cursor(level, page, cursor.0 + 1..)
                        .map(|cursor| (level, cursor))
                })?;
            // Fill in `first_new_level`'s cursor.
            path[first_new_level].1 = first_new_cursor;
            // Fill in the page for the next level down.
            if first_new_level + 1 < CUBOIDS {
                let page = path[first_new_level].0;
                path[first_new_level + 1].0 =
                    self.set.data[usize::try_from(page).unwrap() + usize::from(first_new_cursor.0)];
            }
            first_new_level + 1
        } else {
            0
        };

        // For all the cursors we haven't already filled in, pick the first cursor on
        // the page at that level.
        for level in filled..CUBOIDS {
            let page = path[level].0;
            // If we can't find any cursors at all, that means this set is empty.
            let cursor = self.set.first_cursor(level, page, 0..)?;
            path[level].1 = cursor;
            // Fill in the next level's page.
            if level + 1 < CUBOIDS {
                path[level + 1].0 =
                    self.set.data[usize::try_from(page).unwrap() + usize::from(cursor.0)];
            }
        }

        self.prev_path = Some(path);

        Some(Mapping::new(path.map(|(_, cursor)| cursor)))
    }
}

// i can't be bothered to update this right now
// #[cfg(test)]
// mod tests {
//     use crate::{Cuboid, Mapping, SquareCache};

//     #[test]
//     fn equivalence_classes() {
//         let cuboids = [Cuboid::new(1, 1, 7), Cuboid::new(1, 3, 3)];
//         let square_caches = cuboids.map(SquareCache::new);
//         let equivalence_classes = super::equivalence_classes(cuboids,
// &square_caches);         for class in equivalence_classes {
//             for mapping in &class {
//                 let mapping_data = mapping.to_data(&square_caches);
//                 for equivalent_data in mapping_data.equivalents() {
//                     let equivalent = Mapping::from_data(&square_caches,
// &equivalent_data);                     assert!(class.contains(equivalent));
//                 }
//             }
//         }
//     }
// }
