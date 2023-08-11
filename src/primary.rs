//! The initial home-grown algorithm I came up with.

use std::{
    array,
    collections::HashSet,
    fmt::{self, Display, Formatter, Write},
    fs::{self, File},
    hash::{Hash, Hasher},
    io::{BufReader, BufWriter},
    iter::zip,
    mem,
    path::{Path, PathBuf},
    process::exit,
    sync::{
        atomic::{AtomicUsize, Ordering::Relaxed},
        mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime},
};

use anyhow::{bail, Context};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use serde::{Deserialize, Serialize};

use crate::{
    equivalence_classes, ColoredNet, Cuboid, Direction, Mapping, Net, NetPos, Pos, SkipSet,
    SquareCache, Surface,
};

use Direction::*;

#[derive(Serialize, Deserialize, Clone)]
pub struct NetFinder<const CUBOIDS: usize> {
    /// The cuboids that we're finding common nets for.
    #[serde(with = "crate::utils::arrays")]
    cuboids: [Cuboid; CUBOIDS],
    /// Square caches for the cuboids.
    #[serde(with = "crate::utils::arrays")]
    square_caches: [SquareCache; CUBOIDS],
    /// Mappings that we should skip over because they're the responsibility of
    /// another `NetFinder`.
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
    /// backtrack past this, the `NetFinder` is finished.
    ///
    /// This becomes non-zero when splitting a `NetFinder` in two: if it's
    /// `base_index` instruction hasn't already been backtracked, the existing
    /// `NetFinder` has its `base_index` incremented by one, and it's the new
    /// `NetFinder`'s job to explore all the possibilities where that
    /// instruction isn't run.
    base_index: usize,
    /// The size of the net so far.
    pub area: usize,
    pub target_area: usize,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
enum PosState<const CUBOIDS: usize> {
    /// There's nothing referencing this position in the queue.
    #[default]
    Untouched,
    /// There's an instruction in the queue that wants to map this position to
    /// `cursors`.
    Queued { mapping: Mapping<CUBOIDS> },
    /// There are two instructions in the queue that want to map this position
    /// to different positions. Any further instructions will see this and not
    /// be added.
    ///
    /// `cursors` is the what the first instruction wants this net position to
    /// map to, so that this can be restored to `Queued` if/when the second
    /// instruction gets backtracked.
    Invalid { mapping: Mapping<CUBOIDS> },
    /// This position is filled.
    Filled,
}

impl<const CUBOIDS: usize> NetFinder<CUBOIDS> {
    /// Create a bunch of `NetFinder`s that search for all the nets that fold
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

        // Then create one `NetFinder` per equivalence class.
        let finders = equivalence_classes
            .iter()
            .map(|class| class.canon_mappings().next().unwrap())
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

                pos_states[start_pos] = PosState::Queued {
                    mapping: queue[0].mapping.clone(),
                };

                // Skip all of the equivalence classes prior to this one.
                let mut skip = SkipSet::new(&square_caches);
                for class in &equivalence_classes[..i] {
                    for mapping in class.canon_mappings() {
                        skip.insert(mapping.clone());
                    }
                }

                Self {
                    cuboids: cuboids.clone(),
                    square_caches: square_caches.clone(),
                    skip,

                    queue,
                    potential: Vec::new(),

                    pos_states,
                    surfaces: array::from_fn(|_| Surface::new()),

                    index: 0,
                    base_index: 0,
                    area: 0,
                    target_area: cuboids[0].surface_area(),
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
        let InstructionState::Completed { followup_index } = instruction.state else {
            unreachable!()
        };
        // Mark it as reverted.
        instruction.state = InstructionState::NotRun;
        self.pos_states[instruction.net_pos] = PosState::Queued {
            mapping: instruction.mapping.clone(),
        };
        // Remove the square it added.
        for (surface, cursor) in zip(&mut self.surfaces, instruction.mapping.cursors()) {
            surface.set_filled(cursor.square(), false)
        }
        self.area -= 1;
        // Then remove all the instructions added as a result of this square.
        for instruction in self.queue.drain(followup_index..) {
            // Update the state of the net position the instruction wanted to set.
            self.pos_states[instruction.net_pos] =
                match mem::take(&mut self.pos_states[instruction.net_pos]) {
                    // We've literally just found an instruction that wants to set this position;
                    // this is impossible.
                    PosState::Untouched => unreachable!(),
                    PosState::Queued { .. } => PosState::Untouched,
                    PosState::Invalid { mapping } => PosState::Queued { mapping },
                    // This is an instruction that hasn't been run, so this should be impossible.
                    PosState::Filled => unreachable!(),
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
    #[inline]
    fn handle_instruction(&mut self) {
        let next_index = self.queue.len();
        // Clear `InstructionState::Backtracked` from this instruction if present.
        self.queue[self.index].state = InstructionState::NotRun;
        let instruction = &self.queue[self.index];
        if self.valid(instruction) {
            // Add the new things we can do from here to the queue.
            // Keep track of whether any valid follow-up instructions actually get added.
            let mut followup = false;
            for direction in [Left, Up, Right, Down] {
                let instruction = &self.queue[self.index];
                if let Some(instruction) = self.instruction_in(instruction, direction) {
                    // Look at the status of the spot on the net we're trying to fill.
                    match self.pos_states[instruction.net_pos] {
                        // If it's invalid or already filled, don't add this instruction to the
                        // queue.
                        PosState::Invalid { .. } | PosState::Filled => {}
                        // If it's untouched, queue this instruction and mark it as queued.
                        PosState::Untouched => {
                            self.pos_states[instruction.net_pos] = PosState::Queued {
                                mapping: instruction.mapping.clone(),
                            };
                            // We only consider it a real followup instruction if none of the
                            // squares it sets are already filled.
                            if zip(&self.surfaces, instruction.mapping.cursors())
                                .all(|(surface, cursor)| !surface.filled(cursor.square()))
                            {
                                followup = true;
                            }
                            self.queue.push(instruction);
                        }
                        // If it's already queued to do the same thing as this instruction, this
                        // instruction is redundant. Don't add it to the queue.
                        PosState::Queued { ref mapping } if *mapping == instruction.mapping => {}
                        // If it's queued to do something different, mark it as invalid and queue
                        // this instruction so that it'll become valid again if/when the instruction
                        // is unqueued.
                        PosState::Queued { .. } => {
                            let PosState::Queued { mapping } =
                                mem::take(&mut self.pos_states[instruction.net_pos])
                            else {
                                unreachable!()
                            };
                            self.pos_states[instruction.net_pos] = PosState::Invalid { mapping };
                            self.queue.push(instruction);
                        }
                    }
                }
            }

            // If no follow-up instructions we added, we don't actually fill the square,
            // since we are now considering this a potentially filled square.
            // Since it's possible for it to get invalidated later on, we just mark it as
            // not having been run for now and do a pass through the whole queue to find all
            // the instructions that are still valid, which have to be the ones that we
            // intentionally didn't run here.
            if followup {
                let instruction = &mut self.queue[self.index];
                instruction.state = InstructionState::Completed {
                    followup_index: next_index,
                };
                self.pos_states[instruction.net_pos] = PosState::Filled;
                for (surface, cursor) in zip(&mut self.surfaces, instruction.mapping.cursors()) {
                    surface.set_filled(cursor.square(), true);
                }
                self.area += 1;
            } else {
                // Remove any marker instructions that we added.
                for instruction in self.queue.drain(next_index..) {
                    // Update the state of the net position the instruction wanted to set.
                    self.pos_states[instruction.net_pos] = match std::mem::replace(
                        &mut self.pos_states[instruction.net_pos],
                        PosState::Untouched,
                    ) {
                        PosState::Untouched => unreachable!(),
                        PosState::Queued { .. } => PosState::Untouched,
                        PosState::Invalid { mapping } => PosState::Queued { mapping },
                        PosState::Filled => unreachable!(),
                    }
                }
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
            && matches!(
                self.pos_states[instruction.net_pos],
                PosState::Queued { .. }
            )
    }

    /// Creates a new instruction in a given direction from an instruction, if
    /// the new net position is valid.
    fn instruction_in(
        &self,
        instruction: &Instruction<CUBOIDS>,
        direction: Direction,
    ) -> Option<Instruction<CUBOIDS>> {
        let net_pos = instruction.net_pos.moved_in(direction, &self.pos_states);
        let mut mapping = instruction.mapping.clone();
        zip(&self.square_caches, mapping.cursors_mut())
            .for_each(|(cache, cursor)| *cursor = cursor.moved_in(cache, direction));
        Some(Instruction {
            net_pos,
            mapping,
            state: InstructionState::NotRun,
        })
    }

    /// This method, which should be called when the end of the queue is
    /// reached, goes through all of the unrun instructions to find which ones
    /// are valid, and figures out which combinations of them result in a valid
    /// net.
    ///
    /// It also takes a `search_time` to be inserted into any `Solution`s it
    /// yields.
    fn finalize(&self, prior_search_time: Duration, start: Instant) -> Finalize<'_, CUBOIDS> {
        if self.area + self.potential.len() < self.target_area {
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
            .any(|filled| filled.num_filled() != u32::try_from(self.target_area).unwrap())
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

        // A list of the instructions we know we have to include.
        let mut included = HashSet::new();
        // For each instruction, the list of instructions it conflicts with.
        let mut conflicts: Vec<HashSet<usize>> = vec![HashSet::new(); potential_squares.len()];
        // Go through all the squares on the surfaces of the cuboids to find which
        // instructions we have to include, because they're the only ones that can set a
        // square, as well as which instructions conflict with one another because they
        // set the same surface squares.
        let mut found: Vec<usize> = Vec::new();
        for (cuboid, (cache, surface)) in zip(&self.square_caches, &self.surfaces).enumerate() {
            for square in cache.squares() {
                if !surface.filled(square) {
                    // If the square's not already filled, there has to be at least one potential
                    // square that fills it.
                    found.clear();
                    for (i, instruction) in potential_squares.iter().enumerate() {
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
                        // We already made sure earlier that every unfilled square has at least one
                        // instruction that sets it, so this is impossible.
                        &[] => unreachable!(),
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
        for (i, instruction) in potential_squares.iter().enumerate() {
            for direction in [Left, Up, Down, Right] {
                if let Some(moved_instruction) = self.instruction_in(instruction, direction) {
                    for (j, other_instruction) in potential_squares.iter().enumerate() {
                        if other_instruction.net_pos == moved_instruction.net_pos
                            && other_instruction.mapping != moved_instruction.mapping
                        {
                            conflicts[i].insert(j);
                        }
                    }
                }
            }
        }

        // Now that we've got all the conflicts, make sure that none of the included
        // squares conflict with each other. At the same time, we construct a
        // list of the remaining instructions which aren't guaranteed-included and don't
        // conflict with the included instructions.
        let mut remaining: Vec<_> = (0..potential_squares.len())
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
        let completed: Vec<_> = self
            .queue
            .iter()
            .filter(|instruction| matches!(instruction.state, InstructionState::Completed { .. }))
            .chain(included.iter().map(|&index| &potential_squares[index]))
            .cloned()
            .collect();

        // Calculate how many more squares we still need to fill.
        let remaining_area = self.target_area - self.area - included.len();
        if remaining_area == 0 {
            // If we've already filled all the surface squares, we're done!
            return Finalize::Known(Some(Solution::new(
                &self.square_caches,
                completed.iter(),
                prior_search_time + start.elapsed(),
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
            search_time: prior_search_time + start.elapsed(),

            potential_squares,
            remaining,
            conflicts,
            next: (0..remaining_area).collect(),
        })
    }

    /// Splits this `NetFinder` in place, so that its work is now split between
    /// `self` and the returned `NetFinder`.
    ///
    /// Returns `None` if this can't be done, in which case `self` is unchanged.
    fn split(&mut self) -> Option<Self> {
        let mut new_base_index = self.base_index + 1;
        // Find the first instruction that this `NetFinder` controls which has been run,
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
            // Create the new `NetFinder` by backtracking this one until the instruction at
            // `new_base_index` hasn't been run.
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

/// A solution yielded from `NetFinder`: contains the actual net, as well as the
/// colored versions for each cuboid and the time it was yielded.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Solution {
    /// The solution (canonicalised).
    pub net: Net,
    /// The (canonicalised) colorings of the solution for each cuboid, as
    /// `NetFinder` originally found them.
    pub colored: Vec<ColoredNet>,
    /// The time at which the solution was found.
    pub time: SystemTime,
    /// How long the program had been running for when this solution was found
    /// (including previous runs of the program if the `NetFinder` had been
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

/// The iterator returned by `NetFinder::finalize`.
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

    potential_squares: Vec<Instruction<CUBOIDS>>,
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
                    .map(|&index| &self.potential_squares[self.remaining[index]]),
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
) -> anyhow::Result<impl Iterator<Item = Solution>> {
    let finders = NetFinder::new(cuboids)?;

    Ok(run(cuboids, finders, Vec::new(), Duration::ZERO, progress))
}

#[derive(Serialize, Deserialize)]
pub struct State<const CUBOIDS: usize> {
    pub finders: Vec<NetFinder<CUBOIDS>>,
    pub solutions: Vec<Solution>,
    pub prior_search_time: Duration,
}

/// Updates the passed `state` with the most recently sent `NetFinder`s, then
/// writes it out to a file.
fn update_state<const CUBOIDS: usize>(
    state: &mut State<CUBOIDS>,
    finder_receivers: &mut Vec<Receiver<NetFinder<CUBOIDS>>>,
    channel_rx: &mut Receiver<Receiver<NetFinder<CUBOIDS>>>,
    progress: &ProgressBar,
) {
    state.prior_search_time = progress.elapsed();

    // Check if there are any new `NetFinder`s we need to add to our list.
    loop {
        match channel_rx.try_recv() {
            Err(TryRecvError::Empty) => break,
            Ok(rx) => {
                // There should always be an initial state for the `NetFinder` sent
                // immediately after creating the channel.
                let finder = rx.try_recv().unwrap();
                state.finders.push(finder);
                finder_receivers.push(rx);
                progress.inc_length(1);
            }
            Err(TryRecvError::Disconnected) => {
                // If this was disconnected, the iterator must have been dropped. In that case
                // break from the loop, since we want to write the final state with all the
                // yielded nets.
                break;
            }
        }
    }

    let mut i = 0;
    while i < state.finders.len() {
        match finder_receivers[i].try_recv() {
            Ok(finder) => {
                state.finders[i] = finder;
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                // That `NetFinder` has finished; remove it from our lists.
                state.finders.remove(i);
                finder_receivers.remove(i);
                progress.inc(1);
                // Skip over incrementing `i`, since `i` corresponds to the next
                // `NetFinder` in the list now that we're removed one.
                continue;
            }
        }
        i += 1;
    }

    progress.tick();
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
) -> impl Iterator<Item = Solution> {
    run(
        state.finders[0].cuboids,
        state.finders,
        state.solutions,
        state.prior_search_time,
        progress,
    )
}

/// Runs a `NetFinder` to completion, sending its results and state updates
/// through the provided channels and splitting itself if `current_finders` gets
/// too low.
fn run_finder<'scope, const CUBOIDS: usize>(
    mut finder: NetFinder<CUBOIDS>,
    net_tx: Sender<Solution>,
    finder_tx: SyncSender<NetFinder<CUBOIDS>>,
    channel_tx: Sender<Receiver<NetFinder<CUBOIDS>>>,
    scope: &rayon::Scope<'scope>,
    current_finders: &'scope AtomicUsize,
    prior_search_time: Duration,
    start: Instant,
) {
    let mut send_counter: u16 = 0;
    loop {
        while finder.area < finder.target_area && finder.index < finder.queue.len() {
            // Evaluate the next instruction in the queue.
            finder.handle_instruction();

            send_counter = send_counter.wrapping_add(1);
            if send_counter == 0 {
                let _ = finder_tx.try_send(finder.clone());
                if current_finders.load(Relaxed) < rayon::current_num_threads() + 1 {
                    // Split this `NetFinder` if there aren't currently enough of them to give all
                    // our threads something to do.
                    if let Some(finder) = finder.split() {
                        current_finders.fetch_add(1, Relaxed);

                        // Make a `finder_tx` for the new `NetFinder` to use.
                        let (finder_tx, finder_rx) = mpsc::sync_channel(1);
                        finder_tx.send(finder.clone()).unwrap();
                        channel_tx.send(finder_rx).unwrap();

                        let net_tx = net_tx.clone();
                        let channel_tx = channel_tx.clone();
                        scope.spawn(move |s| {
                            run_finder(
                                finder,
                                net_tx,
                                finder_tx,
                                channel_tx,
                                s,
                                current_finders,
                                prior_search_time,
                                start,
                            )
                        });
                    }
                }
            }
        }

        // We broke out of the loop, which means we've reached the end of the queue or
        // the target area. So, finalize the current net to find solutions and send them
        // off.
        for solution in finder.finalize(prior_search_time, start) {
            net_tx.send(solution).unwrap();
        }

        // Now backtrack and look for more solutions.
        if !finder.backtrack() {
            // Backtracking failed which means there are no solutions left and we're done.
            current_finders.fetch_sub(1, Relaxed);
            return;
        }
    }
}

fn run<const CUBOIDS: usize>(
    cuboids: [Cuboid; CUBOIDS],
    finders: Vec<NetFinder<CUBOIDS>>,
    solutions: Vec<Solution>,
    prior_search_time: Duration,
    progress: ProgressBar,
) -> impl Iterator<Item = Solution> {
    // Create a channel for sending yielded nets to the main thread.
    let (net_tx, net_rx) = mpsc::channel::<Solution>();
    // Create a channel for each `NetFinder` to periodically send its state through.
    let (finder_senders, mut finder_receivers) = finders
        .iter()
        .map(|_| mpsc::sync_channel(1))
        .unzip::<_, _, Vec<_>, Vec<_>>();
    // Then create a channel through which we send the receiving ends of new such
    // channels.
    let (channel_tx, mut channel_rx) = mpsc::channel();

    let current_finders = Arc::new(AtomicUsize::new(finders.len()));
    let start = Instant::now();
    thread::Builder::new()
        .name("scope thread".to_owned())
        .spawn({
            let finders = finders.clone();
            move || {
                rayon::scope(|s| {
                    // Force these to get moved into the closure without moving `current_finders`
                    // into it as well (since then it gets dropped before the scope ends).
                    let net_tx = net_tx;
                    let channel_tx = channel_tx;
                    for (finder, finder_tx) in zip(finders, finder_senders) {
                        let net_tx = net_tx.clone();
                        let channel_tx = channel_tx.clone();
                        s.spawn(|s| {
                            run_finder(
                                finder,
                                net_tx,
                                finder_tx,
                                channel_tx,
                                s,
                                &current_finders,
                                prior_search_time,
                                start,
                            )
                        })
                    }
                });
            }
        })
        .unwrap();

    // Create the folder where we're going to store our state.
    fs::create_dir_all(Path::new(env!("CARGO_MANIFEST_DIR")).join("state")).unwrap();

    progress.set_style(
        ProgressStyle::with_template(
            "{elapsed_precise} {wide_bar} {pos} / {len} finders completed",
        )
        .unwrap(),
    );
    progress.set_length(finders.len().try_into().unwrap());
    progress.set_draw_target(ProgressDrawTarget::stderr());

    // Put the state in a mutex so we can share it with the ctrl+c handler
    let state = Arc::new(Mutex::new(State {
        finders,
        solutions: solutions.clone(),
        prior_search_time: prior_search_time + start.elapsed(),
    }));

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
    let mut yielded_nets: HashSet<Net> = solutions
        .iter()
        .map(|solution| solution.net.clone())
        .collect();

    // Yield all our previous solutions before starting the new ones.
    solutions
        .into_iter()
        .chain(std::iter::from_fn(move || loop {
            update_state(
                &mut state.lock().unwrap(),
                &mut finder_receivers,
                &mut channel_rx,
                &progress,
            );
            // Stop waiting every 50ms to update `state` in case we get Ctrl+C'd.
            match net_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(solution) => {
                    let new = yielded_nets.insert(solution.net.clone());
                    if new {
                        state.lock().unwrap().solutions.push(solution.clone());
                        return Some(solution);
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    // Write out our final state, since it serves as our way of retrieving results
                    // afterwards.
                    write_state(&state.lock().unwrap(), cuboids);
                    return None;
                }
            }
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
