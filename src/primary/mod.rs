//! The initial home-grown algorithm I came up with.

#[cfg(feature = "no-trie")]
use std::simd::{Mask, Simd, SimdPartialEq, SimdUint};
use std::{
    array,
    collections::HashSet,
    fmt::{self, Display, Formatter, Write},
    fs::{self, File},
    hash::{Hash, Hasher},
    io::{BufReader, BufWriter},
    iter::zip,
    num::NonZeroU8,
    path::{Path, PathBuf},
    process::exit,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};

use anyhow::{bail, Context};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use serde::{Deserialize, Serialize};

use crate::{
    equivalence_classes, Class, ClassMapping, ColoredNet, Cuboid, Cursor, Direction, Mapping, Net,
    Pos, SkipSet, Square, SquareCache, Surface,
};

mod cpu;
mod gpu;

use Direction::*;

/// The additional information that we need to have on hand to actually run a
/// `Finder`.
struct FinderCtx<const CUBOIDS: usize> {
    /// The cuboids that we're finding common nets for, from the outside world's
    /// perspective.
    outer_cuboids: [Cuboid; CUBOIDS],
    /// The cuboids that we're finding common nets for, from a `Finder`'s
    /// perspective.
    ///
    /// These are the same cuboids as `outer_cuboids`, but in a different order:
    /// there's always one cuboid on which we start from the same cursor every
    /// time, and this cuboid gets put first. This makes it easier to optimise
    /// how we skip solutions covered by other `Finder`s.
    inner_cuboids: [Cuboid; CUBOIDS],
    /// The common area of all the cuboids in `outer_cuboids` / `inner_cuboids`.
    target_area: usize,

    /// Square caches for the cuboids (`inner_cuboids`, specifically).
    square_caches: [SquareCache; CUBOIDS],
    /// The cursor class on the first cuboid in `inner_cuboids` which we always
    /// use as the starting point on that cuboid.
    #[cfg(feature = "no-trie")]
    fixed_class: Class,
    /// A bitfield indicating whether each cursor on the first cuboid is in the
    /// same family of equivalence classes as `fixed_class`.
    #[cfg(feature = "no-trie")]
    maybe_skipped_lookup: [u64; 4],
    /// The equivalence classes of mappings that we build `Finder`s out of.
    equivalence_classes: Vec<Vec<ClassMapping<CUBOIDS>>>,

    // These are used to timestamp the solutions we get.
    /// How long we've been searching for prior to the start of this run of
    /// `net_finder`.
    prior_search_time: Duration,
    /// The time at which this run started.
    start: Instant,
}

impl<const CUBOIDS: usize> FinderCtx<CUBOIDS> {
    /// Creates a new `FinderCtx` for searching for common nets of the given
    /// cuboids.
    ///
    /// You can also specify how long you've already been searching for, which
    /// will be counted in the timestamps for solutions.
    fn new(cuboids: [Cuboid; CUBOIDS], prior_search_time: Duration) -> anyhow::Result<Self> {
        if cuboids.is_empty() {
            bail!("must specify at least one cuboid");
        }

        if let Some(cuboid) = cuboids[1..]
            .iter()
            .find(|cuboid| cuboid.surface_area() != cuboids[0].surface_area())
        {
            bail!(
                "a {cuboid} cuboid does not have the same surface area as a {} cuboid",
                cuboids[0]
            );
        }

        if cuboids[0].surface_area() > 64 {
            bail!("only cuboids with surface area <= 64 are currently supported");
        }

        let mut square_caches = cuboids.map(SquareCache::new);

        let (fixed_cuboid, fixed_class, equivalence_classes) = equivalence_classes(&square_caches);
        // Move the fixed cuboid to the front of the list.
        let mut inner_cuboids = cuboids;
        inner_cuboids.swap(0, fixed_cuboid);
        // Update the square caches and equivalence classes to also be in the order of
        // the inner cuboids.
        square_caches.swap(0, fixed_cuboid);
        // We can't mutate `HashSet`s in place so we have to create a whole new list for
        // the reordered equivalence classes.
        let mut equivalence_classes: Vec<Vec<ClassMapping<CUBOIDS>>> = equivalence_classes
            .into_iter()
            .map(|class| {
                let mut class: Vec<_> = class
                    .into_iter()
                    .map(|mut mapping| {
                        mapping.classes.swap(0, fixed_cuboid);
                        mapping
                    })
                    .collect();
                // Sort the class by index.
                class.sort_by_key(|mapping| mapping.index());
                class
            })
            .collect();
        // Sort the equivalence classes by the minimum index in each class.
        // That way, each finder knows if a mapping was covered by a previous finder by
        // checking if that mapping's index is less than its.
        equivalence_classes.sort_by_key(|class| class[0].index());

        let mut maybe_skipped_lookup = [0; 4];
        for cursor in square_caches[0]
            .classes()
            .skip(fixed_class.index().into())
            .take(fixed_class.family_size())
            .flat_map(|class| class.contents(&square_caches[0]))
        {
            maybe_skipped_lookup[(cursor.0 >> 6) as usize] |= 1 << (cursor.0 & 0x3f);
        }

        Ok(Self {
            outer_cuboids: cuboids,
            inner_cuboids,
            target_area: cuboids[0].surface_area(),
            square_caches,
            #[cfg(feature = "no-trie")]
            fixed_class,
            #[cfg(feature = "no-trie")]
            maybe_skipped_lookup,
            equivalence_classes,
            prior_search_time,
            start: Instant::now(),
        })
    }

    /// Generates a list of `Finder`s to use for finding the common nets of this
    /// `FinderCtx`'s cuboids.
    fn gen_finders(&self) -> Vec<Finder<CUBOIDS>> {
        // A `SkipSet` containing all of the previous equivalence classes.
        let mut prev_classes = SkipSet::new(self.inner_cuboids);

        self.equivalence_classes
            .iter()
            .map(|class| {
                let start_pos = Pos {
                    x: self.inner_cuboids[0].surface_area().try_into().unwrap(),
                    y: self.inner_cuboids[0].surface_area().try_into().unwrap(),
                };
                let start_class_mapping = class[0];
                let start_mapping = start_class_mapping.sample(&self.square_caches);

                // The first instruction is to add the first square.
                let queue = vec![Instruction {
                    net_pos: start_pos,
                    mapping: start_mapping,
                    followup_index: None,
                }];

                let mut net = [0; 64];
                net[usize::from(start_pos.x % 64)] |= 1 << (start_pos.y % 64);

                // We want to skip all the equivalence classes prior to this one.
                let skip = prev_classes.clone();
                // Then add this equivalence class to the set of previous ones.
                for mapping in class {
                    prev_classes.insert(&self.square_caches, mapping.sample(&self.square_caches));
                }

                Finder {
                    skip,
                    start_mapping_index: start_class_mapping.index(),

                    queue,
                    potential: Vec::new(),

                    net,
                    surfaces: array::from_fn(|_| Surface::new()),

                    index: 0,
                    base_index: 0,
                }
            })
            .collect()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Finder<const CUBOIDS: usize> {
    /// Mappings that we should skip over because they're the responsibility of
    /// another `Finder`.
    skip: SkipSet<CUBOIDS>,
    /// The index of this `Finder`'s starting mapping (as returned by
    /// `ClassMapping::index`).
    start_mapping_index: usize,

    queue: Vec<Instruction<CUBOIDS>>,
    /// The indices of all the 'potential' instructions that we're saving until
    /// the end to run.
    ///
    /// Note that some of these may now be invalid.
    potential: Vec<usize>,

    /// For each column of the net, whether each square in that column has an
    /// instruction trying to set it.
    ///
    /// This is used to deduplicate instructions. That means that if there are
    /// two instructions that map the same net position to different spots on
    /// the surface, only the first one will be added; however, that's actually
    /// a good thing. We don't really want to allow that anyway, since that's
    /// what leads to cuts being required.
    ///
    /// What we should really do in that situation is mark the first one as
    /// invalid too; however it's rare enough that it ends up being faster to
    /// let it slide for now and fix it up at the end.
    ///
    /// Note that this is actually smaller than the net - 64x64, when the net
    /// can go up to 129x129 (with our current max. surface area of 64).
    /// However, this is fine because only a 64x64 region of it can ever be used
    /// - it's only so big because it can be a 64x64 region in either direction.
    ///
    /// So we intentionally allow wrapping around by just using the lower 6 bits
    /// of the coordinates; it doesn't matter that this makes the net
    /// discontiguous, all that matters is that different net positions always
    /// use different bits.
    #[serde(with = "crate::utils::arrays")]
    net: [u64; 64],
    /// Buffers storing which squares we've filled on the surface of each cuboid
    /// so far.
    #[serde(with = "crate::utils::arrays")]
    surfaces: [Surface; CUBOIDS],

    /// The index of the next instruction in `queue` that will be evaluated.
    index: usize,
    /// The index of the first instruction that isn't fixed: when we attempt to
    /// backtrack past this, the `Finder` is finished.
    ///
    /// This becomes non-zero when splitting a `Finder` in two: if it's
    /// `base_index` instruction hasn't already been backtracked, the existing
    /// `Finder` has its `base_index` incremented by one, and it's the new
    /// `Finder`'s job to explore all the possibilities where that
    /// instruction isn't run.
    base_index: usize,
}

/// An instruction to add a square.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// Aligning this to 8 bytes lets the compiler pack it into a 32-bit integer.
#[repr(align(8))]
struct Instruction<const CUBOIDS: usize> {
    /// The position on the net where the square should be added.
    net_pos: Pos,
    /// The cursors on each of the cuboids that that square folds up into.
    mapping: Mapping<CUBOIDS>,
    /// If this instruction has been run, the index in `queue` at which the
    /// instructions added as a result of this instruction begin.
    ///
    /// Otherwise, `None`.
    ///
    /// This will always fit in a `u8` because the queue's length can be at most
    /// 4 times the area of the cuboids, since there can be at most 4
    /// instructions setting each square (1 from each direction). Additionally,
    /// we already limit the area to 64 so that `Cursor` can fit in a `u8`, so
    /// the length is at most 256 and indices are at most 255.
    ///
    /// Also, it can't be 0 because that would imply that this instruction's
    /// index is lower than 0, which makes no sense.
    followup_index: Option<NonZeroU8>,
}

impl<const CUBOIDS: usize> Instruction<CUBOIDS> {
    fn moved_in(&self, ctx: &FinderCtx<CUBOIDS>, direction: Direction) -> Instruction<CUBOIDS> {
        let net_pos = self.net_pos.moved_in_unchecked(direction);
        let mut mapping = self.mapping.clone();
        zip(&ctx.square_caches, mapping.cursors_mut())
            .for_each(|(cache, cursor)| *cursor = cursor.moved_in(cache, direction));
        Instruction {
            net_pos,
            mapping,
            followup_index: None,
        }
    }
}

impl<const CUBOIDS: usize> PartialEq for Instruction<CUBOIDS> {
    fn eq(&self, other: &Self) -> bool {
        self.net_pos == other.net_pos && self.mapping == other.mapping
    }
}

impl<const CUBOIDS: usize> Eq for Instruction<CUBOIDS> {}

impl<const CUBOIDS: usize> Hash for Instruction<CUBOIDS> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.net_pos.hash(state);
        self.mapping.hash(state);
    }
}

impl<const CUBOIDS: usize> Finder<CUBOIDS> {
    /// Undoes the last instruction that was successfully carried out.
    ///
    /// Returns whether backtracking was successful. If `false` is returned,
    /// there are no more available options.
    fn backtrack(&mut self) -> bool {
        // Find the last instruction that was successfully carried out.
        let Some((last_success_index, instruction)) = self
            .queue
            .iter_mut()
            .enumerate()
            .rfind(|(_, instruction)| instruction.followup_index.is_some())
        else {
            return false;
        };
        if last_success_index < self.base_index {
            return false;
        }

        // Mark it as reverted.
        let followup_index: usize = instruction.followup_index.take().unwrap().get().into();
        // Remove the square it added.
        for (surface, cursor) in zip(&mut self.surfaces, instruction.mapping.cursors()) {
            surface.set_filled(cursor.square(), false)
        }
        // Then remove all the instructions added as a result of this square.
        for instruction in self.queue.drain(followup_index..).rev() {
            self.net[usize::from(instruction.net_pos.x % 64)] &=
                !(1 << (instruction.net_pos.y % 64));
        }

        // Also un-mark any instructions that come after this instruction as potential,
        // since they would otherwise have been backtracked first.
        while matches!(self.potential.last(), Some(&index) if index > last_success_index) {
            self.potential.pop();
        }

        // We continue executing from just after the instruction we undid.
        self.index = last_success_index + 1;
        true
    }

    /// Handle the instruction at the current index in the queue, incrementing
    /// the index afterwards.
    fn handle_instruction(&mut self, ctx: &FinderCtx<CUBOIDS>) {
        let instruction = self.queue[self.index];
        // We don't need to check if it's in `self.skip` because we wouldn't have added
        // it to the queue in the first place if that was the case.
        if !zip(&self.surfaces, instruction.mapping.cursors())
            .any(|(surface, cursor)| surface.filled(cursor.square()))
        {
            let old_len = self.queue.len();
            // Add any follow-up instructions.
            let mut neighbours = [Left, Up, Right, Down].map(|direction| Instruction {
                net_pos: instruction.net_pos.moved_in_unchecked(direction),
                // Leave this blank for now so that we can calculate it for all the neighbours at
                // once.
                mapping: Mapping::new([Cursor::new(Square::new(), 0); CUBOIDS]),
                followup_index: None,
            });

            // First, go through each surface and check whether each of this instruction's
            // neighbours try and set squares that are already filled.
            //
            // We do this here so that we don't have to repeatedly load the surfaces inside
            // `add_followup`.
            //
            // We also calculate their mappings while we're at it.
            let mut valid = [true; 4];
            for (cuboid_index, ((cache, surface), cursor)) in ctx
                .square_caches
                .iter()
                .zip(&self.surfaces)
                .zip(instruction.mapping.cursors())
                .enumerate()
            {
                for (neighbour_index, neighbour) in cursor.neighbours(cache).into_iter().enumerate()
                {
                    valid[neighbour_index] &= !surface.filled(neighbour.square());
                    neighbours[neighbour_index].mapping.cursors[cuboid_index] = neighbour;
                }
            }

            for i in 0..4 {
                if valid[i] {
                    self.add_followup(ctx, neighbours[i]);
                }
            }

            // If there are no valid follow-up instructions, we don't actually fill the
            // square, since we are now considering this a potentially filled square.
            // Since it's possible for it to get invalidated later on, we just mark it as
            // not having been run for now and do a pass through the whole queue to find all
            // the instructions that are still valid, which have to be the ones that we
            // intentionally didn't run here.
            if self.queue.len() > old_len {
                let instruction = &mut self.queue[self.index];
                instruction.followup_index =
                    Some(NonZeroU8::new(old_len.try_into().unwrap()).unwrap());
                for (surface, cursor) in zip(&mut self.surfaces, instruction.mapping.cursors()) {
                    surface.set_filled(cursor.square(), true);
                }
            } else {
                // Add this to the list of potential instructions.
                self.potential.push(self.index);
            }
        }
        self.index += 1;
    }

    /// Returns whether an instruction is valid to run.
    fn valid(&self, ctx: &FinderCtx<CUBOIDS>, instruction: &Instruction<CUBOIDS>) -> bool {
        !zip(&self.surfaces, instruction.mapping.cursors())
            .any(|(surface, cursor)| surface.filled(cursor.square()))
            && !self.skip(ctx, instruction.mapping)
    }

    /// Adds the given follow-up instruction to the queue, if it's valid.
    #[inline]
    fn add_followup(&mut self, ctx: &FinderCtx<CUBOIDS>, instruction: Instruction<CUBOIDS>) {
        if !self.skip(ctx, instruction.mapping)
            && self.net[usize::from(instruction.net_pos.x % 64)]
                & (1 << (instruction.net_pos.y % 64))
                == 0
        {
            self.net[usize::from(instruction.net_pos.x % 64)] |= 1 << (instruction.net_pos.y % 64);
            self.queue.push(instruction);
        }
    }

    /// This method, which should be called when the end of the queue is
    /// reached, goes through all of the unrun instructions to find which ones
    /// are valid, and figures out which combinations of them result in a valid
    /// net.
    ///
    /// It also takes a `search_time` to be inserted into any `Solution`s it
    /// yields.
    fn finalize<'a>(&self, ctx: &'a FinderCtx<CUBOIDS>) -> Finalize<'a, CUBOIDS> {
        if self.surfaces[0].num_filled() as usize + self.potential.len() < ctx.target_area {
            // If there aren't at least as many potential instructions as the number of
            // squares left to fill, there's no way that this could produce a valid net.
            return Finalize::Known(None);
        }

        // First we make sure that there's at least one potential square that sets every
        // square on the surface. We do this before actually constructing the
        // list of potential squares because it's where 99% of calls to this function
        // end, so it needs to be fast.
        let mut surfaces = self.surfaces.clone();
        for instruction in self.potential.iter().map(|&index| &self.queue[index]) {
            if self.valid(ctx, instruction) {
                for (surface, cursor) in zip(&mut surfaces, instruction.mapping.cursors()) {
                    surface.set_filled(cursor.square(), true);
                }
            }
        }

        // If either of the two cuboids don't have all their squares filled by either
        // completed or potential instructions, there's no way this will produce a valid
        // net.
        if surfaces
            .into_iter()
            .any(|filled| filled.num_filled() != u32::try_from(ctx.target_area).unwrap())
        {
            return Finalize::Known(None);
        }

        // Now we actually store which instructions are potential squares, which are
        // just all the instructions which would be valid to fill in.
        let potential_squares: Vec<_> = self
            .potential
            .iter()
            .map(|&index| self.queue[index].clone())
            .filter(|instruction| {
                instruction.followup_index.is_none() & self.valid(ctx, instruction)
            })
            .collect();

        let completed = self
            .queue
            .iter()
            .filter(|instruction| instruction.followup_index.is_some())
            .cloned()
            .collect();

        ctx.finalize(completed, potential_squares)
    }

    /// Splits this `Finder` in place, so that its work is now split between
    /// `self` and the returned `Finder`.
    ///
    /// Returns `None` if this can't be done. However, this only happens if
    /// `self` is finished.
    fn split(&mut self, ctx: &FinderCtx<CUBOIDS>) -> Option<Self> {
        // Advance `self` to the end of its queue so that every instruction that can be
        // completed is completed, and we can try to split on them.
        while self.index < self.queue.len() {
            self.handle_instruction(ctx);
        }

        let mut new_base_index = self.base_index + 1;
        // Find the first instruction that this `Finder` controls which has been run,
        // since that's what gets tried first and so that means that the other
        // possibility hasn't been tried yet.
        while self
            .queue
            // Remember, the base index is the index of the first instruction _after_ the ones that
            // are fixed, so the one we're attempting to fix is the one _before_ the base index.
            .get(new_base_index - 1)
            .is_some_and(|instruction| instruction.followup_index.is_none())
        {
            new_base_index += 1;
        }
        if new_base_index - 1 >= self.queue.len() {
            // We couldn't find one, so this can't be split.
            None
        } else {
            // Create the new `Finder` by backtracking this one until the instruction at
            // `new_base_index - 1` hasn't been run.
            let mut new_finder = self.clone();
            while new_finder.queue[new_base_index - 1]
                .followup_index
                .is_some()
            {
                new_finder.backtrack();
            }

            self.base_index = new_base_index;
            new_finder.base_index = new_base_index;

            // Now `self` is responsible for the case where the instruction at
            // `new_base_index` is run, and `new_finder` is responsible for the case where
            // it isn't.
            Some(new_finder)
        }
    }

    /// Returns whether we should skip a mapping because it's already covered by
    /// a previous `Finder`.
    #[inline]
    #[cfg(feature = "no-trie")]
    fn skip(&self, ctx: &FinderCtx<CUBOIDS>, mapping: Mapping<CUBOIDS>) -> bool {
        let index = mapping.cursors[0].0 as usize;
        if (ctx.maybe_skipped_lookup[index >> 6] >> (index & 0x3f)) & 1 == 0 {
            return false;
        }

        // If the fixed class doesn't care about some bits of the transformation, then
        // we're free to alter those bits to make the index of the transformed mapping
        // smaller.
        //
        // To keep track, create a SIMD mask of exactly which transformations are still
        // allowed, and disable some bits as we go along for any transformations which
        // give a non-minimal index.
        let mut valid = Mask::from([true; 8]);

        let transformed: ClassMapping<CUBOIDS> = ClassMapping::new(array::from_fn(|i| {
            let cursor = mapping.cursors[i];
            let cache = &ctx.square_caches[i];
            let mut undo_lookup = Simd::from(cursor.undo_lookup(cache));
            // Set the transforms we get from undoing all the invalid transforms to 8 so
            // they won't show up in `min`.
            undo_lookup = valid.select(undo_lookup, Simd::splat(8));
            // Then find out what the lowest transform we can get is.
            let new_transform = undo_lookup.reduce_min();
            // In order to minimise the index of `transformed`, this is the transform we
            // need to use, since we're going from most-significant to least-significant.
            //
            // So the new set of valid transforms is all the ones that give us this value
            // (and were valid before, remember we already set the invalid ones to 8).
            valid = undo_lookup.simd_eq(Simd::splat(new_transform));
            cursor.class(cache).with_transform(new_transform)
        }));

        transformed.index() < self.start_mapping_index
    }

    /// Returns whether we should skip a mapping because it's already covered by
    /// a previous `Finder`.
    #[cfg(not(feature = "no-trie"))]
    fn skip(&self, _ctx: &FinderCtx<CUBOIDS>, mapping: Mapping<CUBOIDS>) -> bool {
        self.skip.contains(mapping)
    }
}

impl<const CUBOIDS: usize> FinderCtx<CUBOIDS> {
    /// A version of `Finder::finalize` which:
    /// - Does not need an entire `Finder` to be used, just a list of
    ///   instructions.
    /// - Doesn't have the impossible-to-fill-the-surface fast path.
    ///
    /// This is what `Finder::finalize` it calls internally after checking for
    /// that aforementioned fast path, and what's used to process results from
    /// the GPU.
    fn finalize(
        &self,
        completed: Vec<Instruction<CUBOIDS>>,
        mut potential: Vec<Instruction<CUBOIDS>>,
    ) -> Finalize<CUBOIDS> {
        // Check if `completed` is even valid: nothing in `Finder` prevents multiple
        // instructions from setting the same net position, or being neighbours on the
        // net but disagreeing on what each other should map to (which leads to a cut).
        for instruction in completed.iter() {
            let to_check = [
                instruction.clone(),
                instruction.moved_in(self, Left),
                instruction.moved_in(self, Up),
                instruction.moved_in(self, Right),
                instruction.moved_in(self, Down),
            ];
            for other_instruction in completed.iter() {
                if to_check.iter().any(|instruction| {
                    other_instruction.net_pos == instruction.net_pos
                        && other_instruction.mapping != instruction.mapping
                }) {
                    return Finalize::Known(None);
                }
            }
        }

        // Get rid of any potential instructions that are invalid for the same reason.
        potential.retain(|instruction| {
            let to_check = [
                instruction.clone(),
                instruction.moved_in(self, Left),
                instruction.moved_in(self, Up),
                instruction.moved_in(self, Right),
                instruction.moved_in(self, Down),
            ];
            for other_instruction in completed.iter() {
                if to_check.iter().any(|instruction| {
                    other_instruction.net_pos == instruction.net_pos
                        && other_instruction.mapping != instruction.mapping
                }) {
                    return false;
                }
            }
            true
        });

        // First figure out what squares on the surface are already filled by the
        // completed instructions.
        let mut surfaces = [Surface::new(); CUBOIDS];
        for instruction in completed.iter() {
            for (surface, cursor) in zip(&mut surfaces, instruction.mapping.cursors()) {
                surface.set_filled(cursor.square(), true);
            }
        }

        // A list of the instructions we know we have to include.
        let mut included = HashSet::new();
        // For each instruction, the list of instructions it conflicts with.
        let mut conflicts: Vec<HashSet<usize>> = vec![HashSet::new(); potential.len()];
        // Go through all the squares on the surfaces of the cuboids to find which
        // instructions we have to include, because they're the only ones that can set a
        // square, as well as which instructions conflict with one another because they
        // set the same surface squares.
        let mut found: Vec<usize> = Vec::new();
        for (cuboid, (cache, surface)) in zip(&self.square_caches, &surfaces).enumerate() {
            for square in cache.squares() {
                if !surface.filled(square) {
                    // If the square's not already filled, there has to be at least one potential
                    // square that fills it.
                    found.clear();
                    for (i, instruction) in potential.iter().enumerate() {
                        if instruction.mapping.cursors()[cuboid].square() == square {
                            // Note down the conflicts between this instruction and the rest in
                            // `found` so far, then add it to `found`.
                            conflicts[i].extend(found.iter().copied());
                            for j in found.iter().copied() {
                                conflicts[j].insert(i);
                            }
                            found.push(i);
                        }
                    }
                    match found.as_slice() {
                        &[] => return Finalize::Known(None),
                        // If there's only one instruction that fills this square, we have to
                        // include it so that the square gets filled.
                        &[instruction] => {
                            included.insert(instruction);
                        }
                        _ => {}
                    }
                }
            }
        }

        // Two instructions also conflict if they're neighbours on the net, but
        // they disagree on what each other's positions should map to on the
        // surface. To work out when that happens, we go through the neighbours
        // of each instruction's net position and make sure that any instructions that
        // set those positions agree on the face positions they should map to. If they
        // don't, it's a conflict.
        for (i, instruction) in potential.iter().enumerate() {
            for direction in [Left, Up, Down, Right] {
                let moved_instruction = instruction.moved_in(self, direction);
                for (j, other_instruction) in potential.iter().enumerate() {
                    if other_instruction.net_pos == moved_instruction.net_pos
                        && other_instruction.mapping != moved_instruction.mapping
                    {
                        conflicts[i].insert(j);
                    }
                }
            }
        }

        // Now that we've got all the conflicts, make sure that none of the included
        // squares conflict with each other. At the same time, we construct a
        // list of the remaining instructions which aren't guaranteed-included and don't
        // conflict with the included instructions.
        let mut remaining: Vec<_> = (0..potential.len())
            .filter(|instruction| !included.contains(instruction))
            .collect();
        for &instruction in included.iter() {
            for &conflicting in conflicts[instruction].iter() {
                if included.contains(&conflicting) {
                    return Finalize::Known(None);
                }
                if let Ok(index) = remaining.binary_search(&conflicting) {
                    remaining.remove(index);
                }
            }
        }

        // Make a list of all the instructions that have been completed, including
        // `included`.
        let completed: Vec<_> = completed
            .iter()
            .chain(included.iter().map(|&index| &potential[index]))
            .cloned()
            .collect();

        // Calculate how many more squares we still need to fill.
        let remaining_area = self.target_area - completed.len();
        if remaining_area == 0 {
            // If we've already filled all the surface squares, we're done!
            return Finalize::Known(Some(Solution::new(
                &self,
                completed.iter(),
                self.prior_search_time + self.start.elapsed(),
            )));
        }

        if remaining.len() < remaining_area {
            // There aren't enough instructions left to possibly fill the surface.
            return Finalize::Known(None);
        }

        // Finally, we return an iterator which will just brute-force try all the
        // combinations of remaining squares.
        Finalize::Solve(FinishIter {
            ctx: self,
            completed,

            potential,
            remaining,
            conflicts,
            next: (0..remaining_area).collect(),
        })
    }
}

/// A solution yielded from `Finder`: contains the actual net, as well as the
/// colored versions for each cuboid and the time it was yielded.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Solution {
    /// The solution (canonicalised).
    pub net: Net,
    /// The (canonicalised) colorings of the solution for each cuboid, as
    /// `Finder` originally found them.
    pub colored: Vec<ColoredNet>,
    /// The time at which the solution was found.
    pub time: SystemTime,
    /// How long the program had been running for when this solution was found
    /// (including previous runs of the program if the `Finder` had been
    /// resumed, not including the time when the program wasn't running).
    pub search_time: Duration,
}

impl Solution {
    fn new<'a, const CUBOIDS: usize>(
        ctx: &FinderCtx<CUBOIDS>,
        instructions: impl Iterator<Item = &'a Instruction<CUBOIDS>>,
        search_time: Duration,
    ) -> Self {
        let cuboids: Vec<_> = ctx
            .square_caches
            .iter()
            .map(|cache| cache.cuboid())
            .collect();
        let mut net = Net::for_cuboids(&cuboids);
        let mut colored = vec![Net::for_cuboids(&cuboids); cuboids.len()];
        for instruction in instructions {
            net[instruction.net_pos] = true;
            for ((cache, colored_net), cursor) in ctx
                .square_caches
                .iter()
                .zip(&mut colored)
                .zip(instruction.mapping.cursors())
            {
                colored_net[instruction.net_pos] = Some(cursor.square().to_data(cache).face);
            }
        }

        Solution {
            net: net.canon(),
            colored: ctx
                .outer_cuboids
                .map(|cuboid| {
                    let index = ctx
                        .inner_cuboids
                        .iter()
                        .position(|&other_cuboid| other_cuboid == cuboid)
                        .unwrap();
                    colored[index].canon()
                })
                .into(),
            time: SystemTime::now(),
            search_time,
        }
    }
}

/// The iterator returned by `Finder::finalize`.
#[derive(Clone)]
enum Finalize<'a, const CUBOIDS: usize> {
    /// There's a single known solution, or no solution.
    Known(Option<Solution>),
    /// There might be multiple solutions, and it's the inner `FinishIter`'s job
    /// to find them.
    Solve(FinishIter<'a, CUBOIDS>),
}

impl<const CUBOIDS: usize> Iterator for Finalize<'_, CUBOIDS> {
    type Item = Solution;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Finalize::Known(net) => net.take(),
            Finalize::Solve(iter) => iter.next(),
        }
    }
}

/// An iterator which tries all the combinations of the remaining valid
/// potential squares and yields nets of the ones which work.
#[derive(Clone)]
struct FinishIter<'a, const CUBOIDS: usize> {
    ctx: &'a FinderCtx<CUBOIDS>,
    completed: Vec<Instruction<CUBOIDS>>,

    potential: Vec<Instruction<CUBOIDS>>,
    conflicts: Vec<HashSet<usize>>,
    remaining: Vec<usize>,
    /// The next combination to try.
    next: Vec<usize>,
}

impl<const CUBOIDS: usize> FinishIter<'_, CUBOIDS> {
    /// Returns whether the current combination we're trying has conflicts, or
    /// `None` if we've run out of combinations.
    fn has_conflicts(&self) -> Option<bool> {
        if *self.next.last().unwrap() >= self.remaining.len() {
            return None;
        }
        for (i, instruction) in self
            .next
            .iter()
            .map(|&index| self.remaining[index])
            .enumerate()
        {
            for other_instruction in self.next[i + 1..]
                .iter()
                .map(|&index| self.remaining[index])
            {
                if self.conflicts[instruction].contains(&other_instruction) {
                    return Some(true);
                }
            }
        }
        Some(false)
    }

    fn advance(&mut self) {
        // Increment the last index.
        let mut index = self.next.len() - 1;
        self.next[index] += 1;
        while *self.next.last().unwrap() >= self.remaining.len() && index != 0 {
            // If it goes past the end of `remaining`, increment the previous one and then
            // update everything after it to each be 1 more than the previous.
            index -= 1;
            self.next[index] += 1;
            let base = self.next[index] + 1;
            for (i, value) in self.next[index + 1..].iter_mut().enumerate() {
                *value = base + i
            }
        }
    }
}

impl<const CUBOIDS: usize> Iterator for FinishIter<'_, CUBOIDS> {
    type Item = Solution;

    fn next(&mut self) -> Option<Self::Item> {
        while self.has_conflicts()? {
            self.advance();
        }

        // If we got here, we've found a set of instructions which don't conflict.
        // Yield the solution they produce.
        let solution = Solution::new(
            &self.ctx,
            self.completed.iter().chain(
                self.next
                    .iter()
                    .map(|&index| &self.potential[self.remaining[index]]),
            ),
            self.ctx.prior_search_time + self.ctx.start.elapsed(),
        );

        self.advance();

        Some(solution)
    }
}

fn state_path(cuboids: &[Cuboid]) -> PathBuf {
    let mut name = "state/".to_string();
    for (i, cuboid) in cuboids.iter().enumerate() {
        if i != 0 {
            name.push(',');
        }
        write!(name, "{cuboid}").unwrap();
    }
    write!(name, ".json").unwrap();
    Path::new(env!("CARGO_MANIFEST_DIR")).join(name)
}

pub fn find_nets<const CUBOIDS: usize>(
    cuboids: [Cuboid; CUBOIDS],
    progress: ProgressBar,
    gpu: bool,
) -> anyhow::Result<impl Iterator<Item = Solution>> {
    let state = State::new(cuboids)?;
    run(state, progress, gpu)
}

#[derive(Serialize, Deserialize)]
pub struct State<const CUBOIDS: usize> {
    #[serde(with = "crate::utils::arrays")]
    pub cuboids: [Cuboid; CUBOIDS],
    pub finders: Vec<Finder<CUBOIDS>>,
    pub solutions: Vec<Solution>,
    pub prior_search_time: Duration,
}

impl<const CUBOIDS: usize> State<CUBOIDS> {
    /// Creates a new `State` for finding the common nets of the given cuboids
    /// where nothing's been done yet.
    pub fn new(cuboids: [Cuboid; CUBOIDS]) -> anyhow::Result<Self> {
        // TODO we should probably take this as a parameter, right now we wastefully
        // create this twice.
        let ctx = FinderCtx::new(cuboids, Duration::ZERO)?;
        Ok(State {
            cuboids,
            finders: ctx.gen_finders(),
            solutions: Vec::new(),
            prior_search_time: Duration::ZERO,
        })
    }
}

fn write_state<const CUBOIDS: usize>(state: &State<CUBOIDS>, cuboids: [Cuboid; CUBOIDS]) {
    let path = state_path(&cuboids);
    // Initially write to a temporary file so that the previous version is still
    // there if we get Ctrl+C'd while writing or something like that.
    let tmp_path = path.with_extension("json.tmp");
    let file = File::create(&tmp_path).unwrap();
    serde_json::to_writer(BufWriter::new(file), &state).unwrap();
    // Then move it to the real path.
    fs::rename(tmp_path, path).unwrap();
}

pub fn read_state<const CUBOIDS: usize>(
    cuboids: [Cuboid; CUBOIDS],
) -> anyhow::Result<State<CUBOIDS>> {
    let file = File::open(state_path(&cuboids)).context("no state to resume from")?;
    let state = serde_json::from_reader(BufReader::new(file))?;
    Ok(state)
}

pub fn resume<const CUBOIDS: usize>(
    state: State<CUBOIDS>,
    progress: ProgressBar,
    gpu: bool,
) -> anyhow::Result<impl Iterator<Item = Solution>> {
    run(state, progress, gpu)
}

fn run<const CUBOIDS: usize>(
    state: State<CUBOIDS>,
    progress: ProgressBar,
    gpu: bool,
) -> anyhow::Result<impl Iterator<Item = Solution>> {
    let cuboids = state.cuboids;
    let finders = state.finders.clone();
    let solutions = state.solutions.clone();

    // Create the folder where we're going to store our state.
    fs::create_dir_all(Path::new(env!("CARGO_MANIFEST_DIR")).join("state"))?;

    progress.set_style(
        ProgressStyle::with_template(
            "{elapsed_precise} {wide_bar} {pos} / {len} finders completed",
        )
        .unwrap(),
    );
    progress.set_length(finders.len().try_into().unwrap());
    progress.set_draw_target(ProgressDrawTarget::stderr());

    // Put the state in a mutex so we can share it with the ctrl+c handler
    let state = Arc::new(Mutex::new(state));

    ctrlc::set_handler({
        let state = Arc::clone(&state);
        let progress = progress.clone();
        move || {
            // Lock this first so that the main thread doesn't try to keep updating the
            // progress bar.
            let state = state.lock().unwrap();
            progress.finish_and_clear();
            eprintln!("Saving state...");
            write_state(&state, cuboids);
            exit(0);
        }
    })
    .unwrap();

    // Make a set of all the nets (not solutions! we don't care about the colorings
    // and stuff) we've already yielded so that we don't yield duplicates.
    let yielded_nets: HashSet<Net> = solutions
        .iter()
        .map(|solution| solution.net.clone())
        .collect();

    // Yield all our previous solutions before starting the new ones.
    Ok(solutions.into_iter().chain(if gpu {
        Box::new(gpu::run(state, yielded_nets, progress)?) as Box<dyn Iterator<Item = Solution>>
    } else {
        Box::new(cpu::run(state, yielded_nets, progress)?) as Box<dyn Iterator<Item = Solution>>
    }))
}

impl Display for Solution {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut nets = vec![self.net.to_string()];
        for colored_net in self.colored.iter() {
            nets.push(colored_net.to_string());
        }
        write!(f, "{}", join_horizontal(nets))
    }
}

fn join_horizontal(strings: Vec<String>) -> String {
    let mut lines: Vec<_> = strings.iter().map(|s| s.lines()).collect();
    let mut out = String::new();
    loop {
        for (i, iter) in lines.iter_mut().enumerate() {
            if i != 0 {
                out += " ";
            }
            if let Some(line) = iter.next() {
                out += line;
            } else {
                return out;
            }
        }
        out += "\n";
    }
}
