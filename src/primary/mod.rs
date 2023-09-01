//! The initial home-grown algorithm I came up with.

use std::{
    array,
    collections::HashSet,
    fmt::{self, Display, Formatter, Write},
    fs::{self, File},
    hash::{Hash, Hasher},
    io::{BufReader, BufWriter},
    iter::zip,
    path::{Path, PathBuf},
    process::exit,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};

use anyhow::{bail, Context};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use serde::{Deserialize, Serialize};

use crate::{
    equivalence_classes, ColoredNet, Cuboid, Direction, Mapping, Net, NetPos, Pos, SkipSet,
    SquareCache, Surface,
};

mod cpu;
mod gpu;

use Direction::*;

/// The additional information that we need to have on hand to actually run a
/// `Finder`.
struct FinderCtx<const CUBOIDS: usize> {
    /// The cuboids that we're finding common nets for.
    cuboids: [Cuboid; CUBOIDS],
    /// The common area of all the cuboids in `cuboids`.
    target_area: usize,

    /// Square caches for the cuboids.
    square_caches: [SquareCache; CUBOIDS],
    // TODO: move `skip` in here and make it a weird shared franken-trie :).

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

        Ok(Self {
            cuboids,
            target_area: cuboids[0].surface_area(),
            square_caches: cuboids.map(SquareCache::new),
            prior_search_time,
            start: Instant::now(),
        })
    }

    /// The width of the `pos_states` of `Finder`s with this context.
    fn net_width(&self) -> usize {
        2 * self.target_area + 1
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Finder<const CUBOIDS: usize> {
    /// Mappings that we should skip over because they're the responsibility of
    /// another `Finder`.
    skip: SkipSet<CUBOIDS>,

    queue: Vec<Instruction<CUBOIDS>>,
    /// The indices of all the 'potential' instructions that we're saving until
    /// the end to run.
    ///
    /// Note that some of these may now be invalid.
    potential: Vec<usize>,

    /// Information about the state of each position in the net.
    pos_states: Net<PosState<CUBOIDS>>,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Instruction<const CUBOIDS: usize> {
    /// The index in `net.squares` where the square will be added.
    net_pos: NetPos,
    /// The cursors on each of the cuboids that that square folds up into.
    mapping: Mapping<CUBOIDS>,
    state: InstructionState,
}

impl<const CUBOIDS: usize> Instruction<CUBOIDS> {
    fn moved_in(&self, ctx: &FinderCtx<CUBOIDS>, direction: Direction) -> Instruction<CUBOIDS> {
        let net_pos = self.net_pos.moved_in(direction, ctx.net_width());
        let mut mapping = self.mapping.clone();
        zip(&ctx.square_caches, mapping.cursors_mut())
            .for_each(|(cache, cursor)| *cursor = cursor.moved_in(cache, direction));
        Instruction {
            net_pos,
            mapping,
            state: InstructionState::NotRun,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum InstructionState {
    /// An instruction which has not been run, either because:
    /// - We haven't gotten to it yet.
    /// - It's invalid.
    /// - It's been backtracked.
    /// - It's been chosen as a 'potential' instruction to try at the end, which
    ///   doesn't have a separate state because it's possible for it to become
    ///   invalid later and so we'd need an extra check anyway.
    NotRun,
    /// The instruction has been run.
    Completed {
        /// The index in `queue` at which the instructions added as a result of
        /// this instruction begin.
        followup_index: usize,
    },
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

/// The status of a position on the net.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
enum PosState<const CUBOIDS: usize> {
    /// This position could still map to anything.
    #[default]
    Unknown,
    /// If this position gets set, it has to map to the given mapping.
    Known {
        /// The mapping this position is known to map to.
        mapping: Mapping<CUBOIDS>,
        /// The index of the first instruction to require that this position
        /// maps to `mapping`.
        ///
        /// This is *not* the index of the instruction that actually attempts to
        /// map this position to `mapping`; this is the index of its
        /// parent. This is because that instruction might not have been valid,
        /// and so didn't get added to the queue; but we still need to properly
        /// record this `PosState` anyway, and so we instead keep track of the
        /// parent and set this back to `Unknown` once it gets backtracked.
        setter: usize,
    },
    /// This position can't be set because it's required to map to two different
    /// mappings by different neighbouring positions.
    Invalid {
        /// The mapping that `setter` thinks this position should map to.
        mapping: Mapping<CUBOIDS>,
        /// The index of the earliest neighbouring instruction to this position.
        setter: usize,
        /// The index of the first instruction which tried to map this position
        /// to something different from `setter`.
        conflict: usize,
    },
    /// This position is filled.
    Filled {
        /// The mapping that this position maps to.
        mapping: Mapping<CUBOIDS>,
        /// The index of the earliest neighbouring instruction to this position.
        setter: usize,
    },
}

impl<const CUBOIDS: usize> Finder<CUBOIDS> {
    /// Create a bunch of `Finder`s that search for all the nets that fold
    /// into all of the passed cuboids.
    fn new(cuboids: [Cuboid; CUBOIDS]) -> anyhow::Result<Vec<Self>> {
        if cuboids
            .iter()
            .any(|cuboid| cuboid.surface_area() != cuboids[0].surface_area())
        {
            bail!("all passed cuboids must have the same surface area")
        }

        let square_caches = cuboids.map(SquareCache::new);

        // Divide all of the possible starting mappings up several equivalence classes
        // of mappings that will result in the same set of nets.
        let equivalence_classes = equivalence_classes(cuboids, &square_caches);

        // Then create one `Finder` per equivalence class.
        let finders = equivalence_classes
            .iter()
            .map(|class| class.into_iter().next().unwrap())
            .enumerate()
            .map(|(i, start_mapping)| {
                let mut pos_states = Net::for_cuboids(&cuboids);
                let middle = Pos::new(pos_states.width() / 2, pos_states.height() / 2);
                let start_pos = NetPos::from_pos(&pos_states, middle);

                // The first instruction is to add the first square.
                let queue = vec![Instruction {
                    net_pos: start_pos,
                    mapping: start_mapping.clone(),
                    state: InstructionState::NotRun,
                }];

                pos_states[start_pos] = PosState::Known {
                    mapping: queue[0].mapping.clone(),
                    // this is incorrect but it should be fine
                    setter: 0,
                };

                // Skip all of the equivalence classes prior to this one.
                let mut skip = SkipSet::new(cuboids);
                for class in &equivalence_classes[..i] {
                    for mapping in class {
                        skip.insert(&square_caches, mapping);
                    }
                }

                Self {
                    skip,

                    queue,
                    potential: Vec::new(),

                    pos_states,
                    surfaces: array::from_fn(|_| Surface::new()),

                    index: 0,
                    base_index: 0,
                }
            })
            .collect();

        Ok(finders)
    }

    /// Undoes the last instruction that was successfully carried out.
    ///
    /// Returns whether backtracking was successful. If `false` is returned,
    /// there are no more available options.
    fn backtrack(&mut self) -> bool {
        // Find the last instruction that was successfully carried out.
        let Some((last_success_index, instruction)) =
            self.queue.iter_mut().enumerate().rfind(|(_, instruction)| {
                matches!(instruction.state, InstructionState::Completed { .. })
            })
        else {
            return false;
        };
        if last_success_index < self.base_index {
            return false;
        }
        let (InstructionState::Completed { followup_index }, PosState::Filled { mapping, setter }) =
            (instruction.state, self.pos_states[instruction.net_pos])
        else {
            unreachable!()
        };
        // Mark it as reverted.
        instruction.state = InstructionState::NotRun;
        self.pos_states[instruction.net_pos] = PosState::Known { mapping, setter };
        // Remove the square it added.
        for (surface, cursor) in zip(&mut self.surfaces, instruction.mapping.cursors()) {
            surface.set_filled(cursor.square(), false)
        }
        let net_pos = instruction.net_pos;
        // Then remove all the instructions added as a result of this square.
        self.queue.drain(followup_index..);
        // Update the `pos_states` of all the neighbouring positions.
        for direction in [Left, Up, Right, Down] {
            let neighbour = net_pos.moved_in(direction, self.pos_states.width().into());

            // Update the state of the net position the instruction wanted to set.
            match self.pos_states[neighbour] {
                // We've literally just found an instruction next to this position; this is
                // impossible.
                PosState::Unknown => unreachable!(),
                PosState::Known { setter, .. } if setter == last_success_index => {
                    self.pos_states[neighbour] = PosState::Unknown;
                }
                PosState::Invalid {
                    setter,
                    conflict,
                    mapping,
                } if conflict == last_success_index => {
                    self.pos_states[neighbour] = PosState::Known { mapping, setter };
                }
                _ => {}
            }
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
        let next_index = self.queue.len();
        let instruction = &self.queue[self.index];
        // We don't need to check if it's in `self.skip` because we wouldn't have added
        // it to the queue in the first place if that was the case.
        if !zip(&self.surfaces, instruction.mapping.cursors())
            .any(|(surface, cursor)| surface.filled(cursor.square()))
            && !matches!(
                self.pos_states[instruction.net_pos],
                PosState::Invalid { .. }
            )
        {
            // Check whether this instruction has any valid follow-up instructions.
            let has_followups = [Left, Up, Right, Down].into_iter().any(|direction| {
                let instruction = self.queue[self.index].moved_in(ctx, direction);
                matches!(self.pos_states[instruction.net_pos], PosState::Unknown)
                    && !zip(&self.surfaces, instruction.mapping.cursors())
                        .any(|(surface, cursor)| surface.filled(cursor.square()))
                    && !self.skip.contains(instruction.mapping)
            });

            // If there are no valid follow-up instructions, we don't actually fill the
            // square, since we are now considering this a potentially filled square.
            // Since it's possible for it to get invalidated later on, we just mark it as
            // not having been run for now and do a pass through the whole queue to find all
            // the instructions that are still valid, which have to be the ones that we
            // intentionally didn't run here.
            if has_followups {
                let instruction = &mut self.queue[self.index];
                instruction.state = InstructionState::Completed {
                    followup_index: next_index,
                };
                let PosState::Known { setter, mapping } = self.pos_states[instruction.net_pos]
                else {
                    unreachable!()
                };
                self.pos_states[instruction.net_pos] = PosState::Filled { setter, mapping };
                for (surface, cursor) in zip(&mut self.surfaces, instruction.mapping.cursors()) {
                    surface.set_filled(cursor.square(), true);
                }

                // Now that this instruction's properly been run, we update the `pos_states` of
                // its neighbours.
                for direction in [Left, Right, Up, Down] {
                    let instruction = self.queue[self.index].moved_in(ctx, direction);
                    match self.pos_states[instruction.net_pos] {
                        // Neighbouring instructions with unknown mappings now have a known mapping.
                        PosState::Unknown => {
                            self.pos_states[instruction.net_pos] = PosState::Known {
                                mapping: instruction.mapping.clone(),
                                setter: self.index,
                            };
                            if !zip(&self.surfaces, instruction.mapping.cursors())
                                .any(|(surface, cursor)| surface.filled(cursor.square()))
                                && !self.skip.contains(instruction.mapping)
                            {
                                self.queue.push(instruction);
                            }
                        }
                        // If any of this position's neighbours have known mappings that disagree
                        // with what we think they should map to, they're now invalid.
                        PosState::Known { mapping, setter } if mapping != instruction.mapping => {
                            self.pos_states[instruction.net_pos] = PosState::Invalid {
                                mapping,
                                setter,
                                conflict: self.index,
                            }
                        }
                        _ => {}
                    }
                }
            } else {
                // Add this to the list of potential instructions.
                self.potential.push(self.index);
            }
        }
        self.index += 1;
    }

    /// Returns whether an instruction is valid to run.
    fn valid(&self, instruction: &Instruction<CUBOIDS>) -> bool {
        !zip(&self.surfaces, instruction.mapping.cursors())
            .any(|(surface, cursor)| surface.filled(cursor.square()))
            && !self.skip.contains(instruction.mapping)
            && !matches!(
                self.pos_states[instruction.net_pos],
                PosState::Invalid { .. }
            )
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
            if self.valid(instruction) {
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
                matches!(instruction.state, InstructionState::NotRun) & self.valid(instruction)
            })
            .collect();

        ctx.finalize(&self.queue, potential_squares)
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
            .is_some_and(|instruction| {
                !matches!(instruction.state, InstructionState::Completed { .. })
            })
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
            while matches!(
                new_finder.queue[new_base_index - 1].state,
                InstructionState::Completed { .. }
            ) {
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
        completed: &[Instruction<CUBOIDS>],
        potential: Vec<Instruction<CUBOIDS>>,
    ) -> Finalize<CUBOIDS> {
        // First figure out what squares on the surface are already filled by the
        // completed instructions.
        let mut surfaces = [Surface::new(); CUBOIDS];
        for instruction in completed
            .iter()
            .filter(|instruction| matches!(instruction.state, InstructionState::Completed { .. }))
        {
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
            .filter(|instruction| matches!(instruction.state, InstructionState::Completed { .. }))
            .chain(included.iter().map(|&index| &potential[index]))
            .cloned()
            .collect();

        // Calculate how many more squares we still need to fill.
        let remaining_area = self.target_area - completed.len();
        if remaining_area == 0 {
            // If we've already filled all the surface squares, we're done!
            return Finalize::Known(Some(Solution::new(
                &self.square_caches,
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
            square_caches: &self.square_caches,
            completed,
            search_time: self.prior_search_time + self.start.elapsed(),

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
        square_caches: &[SquareCache; CUBOIDS],
        instructions: impl Iterator<Item = &'a Instruction<CUBOIDS>>,
        search_time: Duration,
    ) -> Self {
        let cuboids: Vec<_> = square_caches.iter().map(|cache| cache.cuboid()).collect();
        let mut net = Net::for_cuboids(&cuboids);
        let mut colored = vec![Net::for_cuboids(&cuboids); cuboids.len()];
        for instruction in instructions {
            net[instruction.net_pos] = true;
            for ((cache, colored_net), cursor) in square_caches
                .iter()
                .zip(&mut colored)
                .zip(instruction.mapping.cursors())
            {
                colored_net[instruction.net_pos] = Some(cursor.square().to_data(cache).face);
            }
        }

        Solution {
            net: net.canon(),
            colored: colored.iter().map(Net::canon).collect(),
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
    square_caches: &'a [SquareCache; CUBOIDS],
    completed: Vec<Instruction<CUBOIDS>>,
    search_time: Duration,

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
            &self.square_caches,
            self.completed.iter().chain(
                self.next
                    .iter()
                    .map(|&index| &self.potential[self.remaining[index]]),
            ),
            self.search_time,
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
        Ok(State {
            cuboids,
            finders: Finder::new(cuboids)?,
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
