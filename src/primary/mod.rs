//! The initial home-grown algorithm I came up with.

use std::array;
use std::cmp::Reverse;
use std::collections::{HashSet, VecDeque};
use std::fmt::{self, Debug, Display, Formatter, Write};
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter};
use std::iter::zip;
use std::num::{NonZeroU8, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::process::exit;
#[cfg(feature = "no-trie")]
use std::simd::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{bail, Context};
use arbitrary::Arbitrary;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rustc_hash::FxHashSet;
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
pub struct FinderCtx<const CUBOIDS: usize> {
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
    /// Square caches for the cuboids, in the order of `outer_cuboids`.
    pub outer_square_caches: [SquareCache; CUBOIDS],
    /// The index of the cuboid in `outer_cuboids` which we always use the same
    /// starting point on; in other words, the index of `inner_cuboids[0]` in
    /// `outer_cuboids`.
    pub fixed_cuboid: usize,
    /// The cursor class on the first cuboid in `inner_cuboids` which we always
    /// use as the starting point on that cuboid.
    pub fixed_class: Class,
    /// A bitfield indicating whether each cursor on the first cuboid is in the
    /// same family of equivalence classes as `fixed_class`.
    #[cfg(feature = "no-trie")]
    maybe_skipped_lookup: [u64; 4],
    /// The equivalence classes of mappings that we build `Finder`s out of.
    equivalence_classes: Vec<FxHashSet<ClassMapping<CUBOIDS>>>,

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
    pub fn new(cuboids: [Cuboid; CUBOIDS], prior_search_time: Duration) -> anyhow::Result<Self> {
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

        for i in 0..CUBOIDS {
            if cuboids[i + 1..].contains(&cuboids[i]) {
                bail!("{} cuboid specified twice", cuboids[i])
            }
        }

        if cuboids[0].surface_area() > 64 {
            bail!("only cuboids with surface area <= 64 are currently supported");
        }

        let outer_square_caches = cuboids.map(SquareCache::new);

        let (fixed_cuboid, fixed_class, mut equivalence_classes) =
            equivalence_classes(&outer_square_caches);
        // Move the fixed cuboid to the front of the list.
        // Make sure the remaining cuboids stay in the same order, otherwise mapping
        // indices won't compare the same way anymore (and there'll be a mismatch with
        // the FPGA).
        let mut inner_cuboids = cuboids;
        inner_cuboids[..=fixed_cuboid].rotate_right(1);
        // Update the square caches to also be in the order of the inner cuboids.
        let mut square_caches = outer_square_caches.clone();
        square_caches[..=fixed_cuboid].rotate_right(1);
        // We leave the equivalence classes as is, since they get passed to
        // `Finder::new_blank` which expects things to be in the order of
        // `outer_cuboids` since it's exposed to the outside world (indirectly through
        // `Finder::new`).

        if cfg!(not(feature = "no-trie")) {
            // Sort the equivalence classes in descending order of size, so that more
            // mappings get skipped earlier on.
            equivalence_classes.sort_by_key(|class| Reverse(class.len()));
        }

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
            outer_square_caches,
            fixed_cuboid,
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
    pub fn gen_finders(&self) -> Vec<Finder<CUBOIDS>> {
        // A `SkipSet` containing all of the previous equivalence classes.
        let mut prev_classes = SkipSet::new(self.inner_cuboids);

        self.equivalence_classes
            .iter()
            .map(|class| {
                // We want to skip all the equivalence classes prior to this one.
                let skip = prev_classes.clone();
                // Then add this equivalence class to the set of previous ones.
                for &mapping in class {
                    prev_classes.insert(
                        &self.square_caches,
                        self.to_inner(mapping).sample(&self.square_caches),
                    );
                }

                let mut finder = Finder::new_blank(self, class, skip);
                // Make it so that the first instruction is always run.
                finder.handle_instruction(self);
                finder.base_index = 1;
                finder
            })
            .collect()
    }

    /// Converts a `ClassMapping` whose classes are in the order of
    /// `self.outer_cuboids` to one whose classes are in the order of
    /// `self.inner_cuboids`.
    fn to_inner(&self, mapping: ClassMapping<CUBOIDS>) -> ClassMapping<CUBOIDS> {
        ClassMapping {
            classes: array::from_fn(|inner_index| {
                let cuboid = self.inner_cuboids[inner_index];
                let outer_index = self
                    .outer_cuboids
                    .into_iter()
                    .position(|other_cuboid| other_cuboid == cuboid)
                    .unwrap();
                mapping.classes[outer_index]
            }),
        }
    }

    /// Converts a `ClassMapping` whose classes are in the order of
    /// `self.inner_cuboids` to one whose classes are in the order of
    /// `self.outer_cuboids`.
    fn to_outer(&self, mapping: ClassMapping<CUBOIDS>) -> ClassMapping<CUBOIDS> {
        ClassMapping {
            classes: array::from_fn(|outer_index| {
                let cuboid = self.outer_cuboids[outer_index];
                let inner_index = self
                    .inner_cuboids
                    .into_iter()
                    .position(|other_cuboid| other_cuboid == cuboid)
                    .unwrap();
                mapping.classes[inner_index]
            }),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
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
    ///
    /// Invariant: self.queue[self.base_index] either is past the end of the
    /// queue, hasn't been processed yet, or has been run.
    base_index: usize,
}

/// An almost-losslessly-compressed version of `Finder`, which can be much more
/// easily tested to be valid and as such is capable of implementing
/// `Arbitrary`.
///
/// It's not like it's perfect: the `Arbitrary` implementation can stil spit out
/// `FinderInfo`s with invalid `start_mapping`s and `decisions` which continue
/// after there are no instructions left, but checking those two cases is a lot
/// easier than checking that `Finder`'s million pieces of redundant data all
/// line up.
///
/// Ok, what about the 'almost'? This doesn't encode anything that's been done
/// since the last decision made by the `Finder`, so the index will be back to
/// the index of the instruction that decision was about and `potential` will be
/// back to that state as well. But you can just run a few instructions to get
/// back there.
///
/// This is also the rough format in which `Finder`s are sent to the FPGA. It
/// has the same limitation of not including anything past the last decision
/// when sending its state.
///
/// That's actually quite important: it means that before running `finalize` on
/// the CPU, the `Finder`'s remaining instructions need to be run to get the
/// last few potential instructions ready to go.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FinderInfo<const CUBOIDS: usize> {
    /// The class mapping of the first instruction the finder runs (it can pick
    /// any mapping with those classes, and it will have no effect). The net
    /// position of the instruction has no effect on the `Finder`'s results
    /// so we don't include it.
    pub start_mapping: ClassMapping<CUBOIDS>,
    /// The list of 'decisions' the finder made to get to its current state.
    ///
    /// A decision is whether or not an instruction was run; but we don't
    /// include every instruction, only the ones that were valid to be run (and
    /// potential instructions aren't considered valid for this purpose). So a
    /// decision only ever gets to be 0 if that instruction was backtracked.
    ///
    /// This means we never end up with an invalid decision claiming we have to
    /// run an invalid instruction.
    pub decisions: Vec<bool>,
    /// The index of the first decision in `decisions` that it's this finder's
    /// job to try both options of.
    ///
    /// This can't be 0 because there'd be no reason to ever do that: if you
    /// don't run the first instruction that's it, there's nothing else to run
    /// and so you certainly aren't going to get a solution.
    ///
    /// Enforcing that gets rid of some annoying edge cases on the FPGA.
    pub base_decision: NonZeroUsize,
}

impl<'a, const CUBOIDS: usize> Arbitrary<'a> for FinderInfo<CUBOIDS> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let res = FinderInfo {
            start_mapping: u.arbitrary()?,
            decisions: u.arbitrary()?,
            base_decision: u.arbitrary()?,
        };

        // `base_decision` doesn't actually need to be within `decisions`: it's fine if
        // it's one past the end, since that means that the fixed part is fully
        // specified and it just so happens that nothing's been run past there yet. But
        // if it's any further than that, the fixed part isn't fully specified and it's
        // invalid.
        if res.base_decision.get() > res.decisions.len() {
            return Err(arbitrary::Error::IncorrectFormat);
        }

        // The base decision should always be a 1; if it's a 0, it may as well be moved
        // to the next decision since it can't be backtracked again anyway.
        if res.decisions.get(res.base_decision.get()) == Some(&false) {
            return Err(arbitrary::Error::IncorrectFormat);
        }

        Ok(res)
    }
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
        Instruction {
            net_pos,
            mapping: self.mapping.moved_in(&ctx.square_caches, direction),
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
    /// Creates a new `Finder` from a `FinderInfo`.
    ///
    /// Returns an `Err` if the `FinderInfo` is invalid.
    pub fn new(ctx: &FinderCtx<CUBOIDS>, info: &FinderInfo<CUBOIDS>) -> anyhow::Result<Self> {
        let (class_index, class) = ctx
            .equivalence_classes
            .iter()
            .enumerate()
            .find(|(_, class)| class.contains(&info.start_mapping))
            // This will also catch situations where the classes of `info.start_mapping` are out of
            // range.
            .context("invalid start_mapping")?;

        let mut skip = SkipSet::new(ctx.inner_cuboids);
        for class in &ctx.equivalence_classes[..class_index] {
            for &mapping in class {
                skip.insert(
                    &ctx.square_caches,
                    ctx.to_inner(mapping).sample(&ctx.square_caches),
                );
            }
        }

        let mut this = Finder::new_blank(ctx, class, skip);

        let mut decision_index = 0;
        let mut base_index = None;
        while decision_index < info.decisions.len() {
            if this.index >= this.queue.len() {
                bail!("Ran out of instructions to run when there were still decisions left");
            }
            let index = this.index;
            let prev_area = this.area();
            this.handle_instruction(ctx);
            if this.area() > prev_area {
                // It just ran the instruction, so clearly the instruction was valid to run and
                // this is a decision: if the decision was meant to be not running it, just
                // backtrack.
                if !info.decisions[decision_index] {
                    // Reset this to make sure we can backtrack.
                    this.base_index = 0;
                    assert!(this.backtrack());
                }
                if decision_index == info.base_decision.get() {
                    // We've just found what index in the queue `base_decision` corresponds to; set
                    // `base_index` to that.
                    base_index = Some(index);
                }
                decision_index += 1;
            }
        }

        // If `base_index` is `None`, that means `base_decision` is past the end of
        // `decisions`, which means that all the decisions are fixed but everything from
        // there on can change.
        //
        // And we've just gone through all the decisions, so `this.index` is the index
        // of the first instruction not covered by a decision.
        this.base_index = base_index.unwrap_or(this.index);

        Ok(this)
    }

    /// Creates a new, blank `Finder` from a `SkipSet` and an equivalence class
    /// of mappings to pick its starting position from.
    ///
    /// The `SkipSet` is an optimisation, it takes a while to construct them and
    /// it's significantly faster to build up one `SkipSet` that we repeatedly
    /// copy as we go along instead of making a whole new one for every
    /// `Finder`.
    fn new_blank(
        ctx: &FinderCtx<CUBOIDS>,
        class: &FxHashSet<ClassMapping<CUBOIDS>>,
        skip: SkipSet<CUBOIDS>,
    ) -> Self {
        let start_pos = Pos {
            x: ctx.inner_cuboids[0].surface_area().try_into().unwrap(),
            y: ctx.inner_cuboids[0].surface_area().try_into().unwrap(),
        };

        // We start from the mapping with the smallest index, since `skip` relies on it.
        //
        // Note that the one with the smallest index always uses the correct fixed class
        // since we always pick the root of the family to be the fixed class.
        let start_class_mapping = *class
            .iter()
            .min_by_key(|&&mapping| ctx.to_inner(mapping).index())
            .unwrap();
        // `equivalence_classes` uses the outer order of cuboids, so we need to
        // translate.
        let inner_start_class_mapping = ctx.to_inner(start_class_mapping);
        let start_mapping = inner_start_class_mapping.sample(&ctx.square_caches);

        // The first instruction is to add the first square.
        let queue = vec![Instruction {
            net_pos: start_pos,
            mapping: start_mapping,
            followup_index: None,
        }];

        let mut net = [0; 64];
        net[usize::from(start_pos.x % 64)] |= 1 << (start_pos.y % 64);

        Finder {
            skip,
            start_mapping_index: inner_start_class_mapping.index(),

            queue,
            potential: Vec::new(),

            net,
            surfaces: array::from_fn(|_| Surface::new()),

            index: 0,
            base_index: 0,
        }
    }

    /// Returns the `FinderInfo` of this `Finder`.
    ///
    /// This method takes ownership of `self` because it needs to mutate
    /// `self`'s state and recreate the state at the time an instruction was
    /// processed to figure out whether not running it was a decision.
    pub fn into_info(mut self, ctx: &FinderCtx<CUBOIDS>) -> FinderInfo<CUBOIDS> {
        let inner_start_mapping = self.queue[0].mapping.to_classes(&ctx.square_caches);
        // Reorder `inner_start_mapping` to be in the order of `ctx.outer_cuboids`, not
        // `ctx.inner_cuboids`.
        let start_mapping = ctx.to_outer(inner_start_mapping);

        let base_index = self.base_index;

        let mut decisions = VecDeque::new();
        let mut base_decision = None;

        // Go through all the instructions we've processed and, if they're decisions,
        // add them to the list.
        for index in (0..self.index).rev() {
            let instruction = &self.queue[index];
            // Whether to increment `base_decision`, if it's `Some`.
            let mut inc_base_decision = false;
            if instruction.followup_index.is_some() {
                // This instruction's been run, so clearly it was a decision.
                decisions.push_front(true);
                // Reset the base index so that we can backtrack whatever we want.
                self.base_index = 0;
                assert!(self.backtrack());
                if index == base_index {
                    // We found the base decision. For the moment, its index is 0, and we'll add to
                    // it as we add more decisions.
                    base_decision = Some(0)
                } else {
                    // Only increment `base_decision` if it was already `Some`, not if we just set
                    // it - it's still index 0 at that point.
                    inc_base_decision = true;
                }
            } else {
                // `base_index` should always be the index of an instruction that was run.
                debug_assert_ne!(index, base_index);
                // Figure out if not running this instruction was a decision by trying to run
                // it.
                self.index = index;
                let old_area = self.area();
                self.handle_instruction(ctx);
                if self.area() > old_area {
                    // It got run, so that means it's valid and not running it was a decision.
                    self.base_index = 0;
                    assert!(self.backtrack());
                    decisions.push_front(false);
                    inc_base_decision = true;
                } else {
                    // Otherwise it's not a decision, so we do nothing.
                }
            }

            if let Some(base_decision) = &mut base_decision {
                if inc_base_decision {
                    *base_decision += 1;
                }
            }
        }

        FinderInfo {
            start_mapping,
            // The only scenario where `base_decision` can be `None` here is if `base_index` is past
            // the end of the queue, meaning that the finder is responsible for doing... absolutely
            // nothing. There are no instructions for it to try and run.
            //
            // While pointless, that's still a valid situation and maps to `base_decision` being 1
            // past the end of the list of decisions, meaning that all the current decisions are
            // fixed and the only thing the finder can do is add more. In this case, it turns out it
            // actually can't add more but there's no way nor need to encode that.
            base_decision: base_decision
                .unwrap_or(decisions.len())
                .try_into()
                .expect("finder with base_index 0"),
            decisions: decisions.into(),
        }
    }

    /// Runs this `Finder` until its decisions have changed.
    ///
    /// Returns a tuple of `(solution, success)`.
    ///
    /// If backtracking was the cause of the change in decisions, `solution` is
    /// a `FinderInfo` of the finder's state prior to backtracking if that state
    /// was a possible solution. Otherwise, it's `None`.
    ///
    /// `success` indicates whether stepping was successful; if it's `false`,
    /// the finder is done.
    pub fn step(&mut self, ctx: &FinderCtx<CUBOIDS>) -> (Option<FinderInfo<CUBOIDS>>, bool) {
        let start_area = self.area();
        let mut solution = None;
        while self.area() == start_area {
            if self.index < self.queue.len() {
                self.handle_instruction(ctx);
            } else {
                solution = self
                    .possible_solution(ctx)
                    .then(|| self.clone().into_info(ctx));
                if !self.backtrack() {
                    return (solution, false);
                }
            }
        }
        (solution, true)
    }

    /// Returns the number of instructions in the queue that are currently run;
    /// in other words, the number of squares filled on the surfaces so far.
    pub fn area(&self) -> u8 {
        self.surfaces[0].num_filled().try_into().unwrap()
    }

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
        if last_success_index == self.base_index {
            // For the same reasons as in `handle_instruction`, we want to increment
            // `base_index` if it now points to a 0.
            self.base_index += 1;
        }

        true
    }

    /// Handle the instruction at the current index in the queue, incrementing
    /// the index afterwards.
    fn handle_instruction(&mut self, ctx: &FinderCtx<CUBOIDS>) {
        let instruction = self.queue[self.index];
        let mut run = false;
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
                run = true;
            } else {
                // Add this to the list of potential instructions.
                self.potential.push(self.index);
            }
        }

        if !run && self.index == self.base_index {
            // If we didn't run the instruction at `base_index`, increment it.
            //
            // It makes no difference to the behaviour since all the base index means is
            // that you're allowed to backtrack instructions from here on, and this
            // instruction now can't be backtracked anyway. So really, you can now only
            // backtrack instructions from `base_index + 1` on.
            //
            // Doing this means that there's only one canonical `base_index` for a finder,
            // which is important for fuzzing purposes.
            self.base_index += 1;
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

    /// Returns whether taking the current state of this `Finder` and possibly
    /// running some of its potential instructions might lead to a solution.
    ///
    /// It's not guaranteed this being `true` means there'll be a solution, but
    /// it returns `false` in all the most common cases of there not being.
    pub fn possible_solution(&self, ctx: &FinderCtx<CUBOIDS>) -> bool {
        if self.area() as usize + self.potential.len() < ctx.target_area {
            // If there aren't at least as many potential instructions as the number of
            // squares left to fill, there's no way that this could produce a valid net.
            return false;
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
            return false;
        }

        true
    }

    /// This method, which should be called when the end of the queue is
    /// reached, goes through all of the unrun instructions to find which ones
    /// are valid, and figures out which combinations of them result in a valid
    /// net.
    ///
    /// It also takes a `search_time` to be inserted into any `Solution`s it
    /// yields.
    fn finalize<'a>(&self, ctx: &'a FinderCtx<CUBOIDS>) -> Finalize<'a, CUBOIDS> {
        if !self.possible_solution(ctx) {
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
    pub fn split(&mut self, ctx: &FinderCtx<CUBOIDS>) -> Option<Self> {
        // Try and run the instruction at `self.base_index` so that we can split on it.
        while self.index <= self.base_index && self.index < self.queue.len() {
            // Note that this will also increment `self.base_index` if that instruction
            // happens to be invalid.
            self.handle_instruction(ctx);
        }

        if self.base_index >= self.queue.len() {
            // `self.base_index` is past the end of the queue, so we can't split on it.
            None
        } else {
            // Create the new `Finder` by backtracking this one until the instruction at
            // `self.base_index` hasn't been run.
            let mut new_finder = self.clone();
            while new_finder.queue[self.base_index].followup_index.is_some() {
                new_finder.backtrack();
            }

            debug_assert_eq!(new_finder.index, self.base_index + 1);
            new_finder.base_index = self.base_index + 1;

            // Move `self.base_index` forwards to the first run instruction in
            // `self.queue[self.base_index + 1..]`.
            self.base_index = self
                .queue
                .iter()
                .enumerate()
                .filter(|&(i, _)| i > self.base_index)
                .find(|(_, instruction)| instruction.followup_index.is_some())
                .map(|(i, _)| i)
                // Or if there isn't one, the instruction we're going to process next.
                .unwrap_or(self.index);

            // Now `self` is responsible for the case where the instruction at the old value
            // of `self.base_index` is run, and `new_finder` is responsible for the case
            // where it isn't.
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

impl<const CUBOIDS: usize> Debug for Finder<CUBOIDS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        struct QueueDebug<'a, const CUBOIDS: usize>(&'a Finder<CUBOIDS>);
        impl<const CUBOIDS: usize> Debug for QueueDebug<'_, CUBOIDS> {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                let mut d = f.debug_list();
                for (index, instruction) in self.0.queue.iter().enumerate() {
                    let base_char = if index == self.0.base_index { 'b' } else { ' ' };
                    let target_char = if index == self.0.index { '>' } else { ' ' };
                    let pos = instruction.net_pos;
                    let mapping = instruction.mapping;
                    let followup = instruction
                        .followup_index
                        .map_or(String::new(), |index| format!(" fu {index}"));
                    d.entry(&format_args!(
                        "{base_char}{target_char} {index}: {pos} -> {mapping}{followup}"
                    ));
                }
                d.finish()
            }
        }

        f.debug_struct("Finder")
            .field("queue", &QueueDebug(self))
            .finish()
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
