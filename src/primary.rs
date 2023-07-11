//! The initial home-grown algorithm I came up with.

use std::{
    collections::HashSet,
    fmt::Write,
    fs::{self, File},
    hash::{Hash, Hasher},
    io::{BufReader, BufWriter},
    iter::zip,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError},
    thread,
    time::Duration,
};

use anyhow::{bail, Context};
use rayon::prelude::ParallelIterator;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};
use spliter::{ParallelSpliterator, Spliterator};

use crate::{Cuboid, CursorData, Direction, Mapping, MappingData, Net, Square, SquareCache};

use Direction::*;

#[derive(Serialize, Deserialize, Clone)]
pub struct NetFinder {
    /// The cuboids that we're finding common nets for.
    cuboids: [Cuboid; 2],
    /// Square caches for the cuboids.
    square_caches: [SquareCache; 2],
    /// Mappings that we should skip over because they're the responsibility of
    /// another `NetFinder`.
    skip: FxHashSet<Mapping>,

    queue: Vec<Instruction>,
    /// The indices of all the 'potential' instructions that we're saving until
    /// the end to run.
    ///
    /// Note that some of these may now be invalid.
    potential: Vec<usize>,

    pub net: Net,
    /// Information about the state of each position in the net.
    pos_states: Vec<PosState>,
    /// A bitmask of which squares are filled on the surface of each cuboid.
    filled: [u64; 2],

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
struct Instruction {
    /// The index in `net.squares` where the square will be added.
    net_pos: IndexPos,
    /// The cursors on each of the cuboids that that square folds up into.
    mapping: Mapping,
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

impl PartialEq for Instruction {
    fn eq(&self, other: &Self) -> bool {
        self.net_pos == other.net_pos && self.mapping == other.mapping
    }
}

impl Eq for Instruction {}

impl Hash for Instruction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.net_pos.hash(state);
        self.mapping.hash(state);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct IndexPos(usize);

impl IndexPos {
    /// Moves this position in `direction` on a given net.
    fn moved_in(self, direction: Direction, net: &Net) -> Option<Self> {
        let new_index = match direction {
            Left => self.0 - 1,
            Up => self.0 - usize::from(net.width),
            Right => self.0 + 1,
            Down => self.0 + usize::from(net.width),
        };
        if new_index >= net.squares.len() {
            None
        } else {
            Some(Self(new_index))
        }
    }
}

/// The status of a position on the net.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum PosState {
    /// There's nothing referencing this position in the queue.
    Untouched,
    /// There's an instruction in the queue that wants to map this position to
    /// `cursors`.
    Queued { mapping: Mapping },
    /// There are two instructions in the queue that want to map this position
    /// to different positions. Any further instructions will see this and not
    /// be added.
    ///
    /// `cursors` is the what the first instruction wants this net position to
    /// map to, so that this can be restored to `Queued` if/when the second
    /// instruction gets backtracked.
    Invalid { mapping: Mapping },
    /// This position is filled.
    Filled,
}

impl NetFinder {
    /// Create a bunch of `NetFinder`s that search for all the nets that fold
    /// into all of the passed cuboids.
    fn new(cuboids: [Cuboid; 2]) -> anyhow::Result<Vec<Self>> {
        if cuboids
            .iter()
            .any(|cuboid| cuboid.surface_area() != cuboids[0].surface_area())
        {
            bail!("all passed cuboids must have the same surface area")
        }

        let net = Net::for_cuboids(&cuboids);
        let middle_x = net.width() / 2;
        let middle_y = net.height() / 2;

        let square_caches = cuboids.map(SquareCache::new);

        // Divide all of the possible starting mappings up several equivalence classes
        // of mappings that will result in the same set of nets.
        let mut equivalence_classes: Vec<FxHashSet<Mapping>> = Vec::new();

        for cursor in cuboids[1].unique_cursors() {
            let mapping_data = MappingData::new(CursorData::new(cuboids[0]), cursor);
            let mapping = Mapping::from_data(&square_caches, &mapping_data);

            if !equivalence_classes
                .iter()
                .any(|class| class.contains(&mapping))
            {
                // We've found a mapping that's in a new equivalence class. Add it to the list.
                equivalence_classes.push(
                    mapping_data
                        .equivalents()
                        .into_iter()
                        .map(|mapping| Mapping::from_data(&square_caches, &mapping))
                        .collect(),
                )
            }
        }

        // Then create one `NetFinder` per equivalence class.
        let finders = equivalence_classes
            .iter()
            .map(|class| class.iter().next().unwrap())
            .enumerate()
            .map(|(i, &start_mapping)| {
                // The first instruction is to add the first square.
                let start_pos = IndexPos(
                    usize::from(middle_y) * usize::from(net.width()) + usize::from(middle_x),
                );
                let queue = vec![Instruction {
                    net_pos: start_pos,
                    mapping: start_mapping,
                    state: InstructionState::NotRun,
                }];
                let mut pos_states = vec![PosState::Untouched; net.squares.len()];
                pos_states[start_pos.0] = PosState::Queued {
                    mapping: queue[0].mapping,
                };
                // Skip all of the equivalence classes prior to this one.
                let skip = equivalence_classes[..i]
                    .iter()
                    .flat_map(|class| class.iter().copied())
                    .collect();
                Self {
                    cuboids,
                    square_caches: square_caches.clone(),
                    skip,

                    queue,
                    potential: Vec::new(),

                    net: net.clone(),
                    pos_states,
                    filled: [0; 2],

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
        self.pos_states[instruction.net_pos.0] = PosState::Queued {
            mapping: instruction.mapping,
        };
        // Remove the square it added.
        self.set_square(last_success_index, false);
        // Then remove all the instructions added as a result of this square.
        for instruction in self.queue.drain(followup_index..) {
            // Update the state of the net position the instruction wanted to set.
            self.pos_states[instruction.net_pos.0] = match self.pos_states[instruction.net_pos.0] {
                // We've literally just found an instruction that wants to set this position; this
                // is impossible.
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
                    match self.pos_states[instruction.net_pos.0] {
                        // If it's invalid or already filled, don't add this instruction to the
                        // queue.
                        PosState::Invalid { .. } | PosState::Filled => {}
                        // If it's untouched, queue this instruction and mark it as queued.
                        PosState::Untouched => {
                            self.pos_states[instruction.net_pos.0] = PosState::Queued {
                                mapping: instruction.mapping,
                            };
                            // We only consider it a real followup instruction if none of the
                            // squares it sets are already filled.
                            if instruction
                                .mapping
                                .cursors
                                .iter()
                                .enumerate()
                                .all(|(cuboid, cursor)| !self.filled(cuboid, cursor.square()))
                            {
                                followup = true;
                            }
                            self.queue.push(instruction);
                        }
                        // If it's already queued to do the same thing as this instruction, this
                        // instruction is redundant. Don't add it to the queue.
                        PosState::Queued { mapping } if mapping == instruction.mapping => {}
                        // If it's queued to do something different, mark it as invalid and queue
                        // this instruction so that it'll become valid again if/when the instruction
                        // is unqueued.
                        PosState::Queued { mapping } => {
                            self.pos_states[instruction.net_pos.0] = PosState::Invalid { mapping };
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
                self.pos_states[instruction.net_pos.0] = PosState::Filled;
                self.set_square(self.index, true);
            } else {
                // Remove any marker instructions that we added.
                for instruction in self.queue.drain(next_index..) {
                    // Update the state of the net position the instruction wanted to set.
                    self.pos_states[instruction.net_pos.0] =
                        match self.pos_states[instruction.net_pos.0] {
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
    fn valid(&self, instruction: &Instruction) -> bool {
        !self.filled(0, instruction.mapping.cursors[0].square())
            & !self.filled(1, instruction.mapping.cursors[1].square())
            & !self.skip.contains(&instruction.mapping)
            & matches!(
                self.pos_states[instruction.net_pos.0],
                PosState::Queued { .. }
            )
    }

    /// Returns whether a square on the surface of the given cuboid is filled.
    fn filled(&self, cuboid: usize, square: Square) -> bool {
        self.filled[cuboid] & (1 << square.0) != 0
    }

    /// Sets whether a square on the surface of the given cuboid is filled.
    fn set_filled(&mut self, cuboid: usize, square: Square, value: bool) {
        let mask = 1 << square.0;
        self.filled[cuboid] &= !mask;
        self.filled[cuboid] |= mask * u64::from(value);
    }

    /// Creates a new instruction in a given direction from an instruction, if
    /// the new net position is valid.
    fn instruction_in(
        &self,
        instruction: &Instruction,
        direction: Direction,
    ) -> Option<Instruction> {
        let net_pos = instruction.net_pos.moved_in(direction, &self.net)?;
        // It's easier to just mutate the array in place than turn it into an iterator
        // and back.
        let mut mapping = instruction.mapping;
        zip(&self.square_caches, &mut mapping.cursors)
            .for_each(|(cache, cursor)| *cursor = cursor.moved_in(cache, direction));
        Some(Instruction {
            net_pos,
            mapping,
            state: InstructionState::NotRun,
        })
    }

    /// Set a square on the net and surfaces of the cuboids simultaneously,
    /// specifying the square by the index of an instruction which sets it.
    #[inline]
    fn set_square(&mut self, instruction_index: usize, value: bool) {
        let Instruction {
            net_pos, mapping, ..
        } = self.queue[instruction_index];
        match (self.net.squares[net_pos.0], value) {
            (false, true) => self.area += 1,
            (true, false) => self.area -= 1,
            _ => {}
        }
        self.net.squares[net_pos.0] = value;
        for (cuboid, cursor) in mapping.cursors.into_iter().enumerate() {
            self.set_filled(cuboid, cursor.square(), value)
        }
    }

    /// This method, which should be called when the end of the queue is
    /// reached, goes through all of the unrun instructions to find which ones
    /// are valid, and figures out which combinations of them result in a valid
    /// net.
    fn finalize(&self) -> Finalize {
        if self.area + self.potential.len() < self.target_area {
            // If there aren't at least as many potential instructions as the number of
            // squares left to fill, there's no way that this could produce a valid net.
            return Finalize::Known(None);
        }

        // First we make sure that there's at least one potential square that sets every
        // square on the surface. We do this before actually constructing the
        // list of potential squares because it's where 99% of calls to this function
        // end, so it needs to be fast.
        // Use a bitfield to store which squares are filled in on each cuboid, starting
        // off with the ones that are already properly filled in.
        let mut filled: [u64; 2] = self.filled;
        for instruction in self.potential.iter().map(|&index| &self.queue[index]) {
            if self.valid(instruction) {
                for (filled, cursor) in zip(&mut filled, instruction.mapping.cursors) {
                    *filled |= 1 << cursor.square().0;
                }
            }
        }

        // If either of the two cuboids don't have all their squares filled by either
        // completed or potential instructions, there's no way this will produce a valid
        // net.
        if filled
            .into_iter()
            .any(|filled| filled.trailing_ones() != u32::try_from(self.target_area).unwrap())
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
        for cuboid in 0..2 {
            for square in self.square_caches[cuboid].squares() {
                if !self.filled(cuboid, square) {
                    // If the square's not already filled, there has to be at least one potential
                    // square that fills it.
                    found.clear();
                    for (i, instruction) in potential_squares.iter().enumerate() {
                        if instruction.mapping.cursors[cuboid].square() == square {
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

        // Make a net with all the included instructions filled in.
        let mut net = self.net.clone();
        for instruction in included.iter().map(|&index| &potential_squares[index]) {
            net.squares[instruction.net_pos.0] = true;
        }

        // Calculate how many more squares we still need to fill.
        let remaining_area = self.target_area - self.area - included.len();
        if remaining_area == 0 {
            // If we've already filled all the surface squares, we're done!
            return Finalize::Known(Some(net));
        }

        if remaining.len() < remaining_area {
            // There aren't enough instructions left to possibly fill the surface.
            return Finalize::Known(None);
        }

        // Finally, we return an iterator which will just brute-force try all the
        // combinations of remaining squares.
        Finalize::Solve(FinishIter {
            net,
            potential_squares,
            remaining,
            conflicts,
            next: (0..remaining_area).collect(),
        })
    }
}

/// The iterator returned by `NetFinder::finalize`.
#[derive(Clone)]
enum Finalize {
    /// There's a single known solution, or no solution.
    Known(Option<Net>),
    /// There might be multiple solutions, and it's the inner `FinishIter`'s job
    /// to find them.
    Solve(FinishIter),
}

impl Iterator for Finalize {
    type Item = Net;

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
struct FinishIter {
    net: Net,
    potential_squares: Vec<Instruction>,
    conflicts: Vec<HashSet<usize>>,
    remaining: Vec<usize>,
    /// The next combination to try.
    next: Vec<usize>,
}

impl FinishIter {
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

impl Iterator for FinishIter {
    type Item = Net;

    fn next(&mut self) -> Option<Self::Item> {
        while self.has_conflicts()? {
            self.advance();
        }
        // If we got here, we've found a set of instructions which don't conflict.
        // Fill in their squares on the net and yield it.
        let mut net = self.net.clone();
        for instruction in self
            .next
            .iter()
            .map(|&index| &self.potential_squares[self.remaining[index]])
        {
            net.squares[instruction.net_pos.0] = true;
        }

        self.advance();

        Some(net)
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

/// Sorts cuboids first in descending order of how many of their surface squares
/// we have to consider, since we only have to consider 1 on the first one and
/// so we want to pick the biggest number of them to eliminate, and then in
/// ascending lexicographic order of their components if two have the same
/// number of surface squares to consider.
fn sort_cuboids(cuboids: &mut [Cuboid]) {
    cuboids.sort_by(|a, b| {
        b.unique_cursors()
            .len()
            .cmp(&a.unique_cursors().len())
            .then(a.cmp(b))
    });
}

pub fn find_nets(mut cuboids: [Cuboid; 2]) -> anyhow::Result<impl Iterator<Item = Net>> {
    sort_cuboids(&mut cuboids);
    let finders = NetFinder::new(cuboids)?;

    Ok(run(cuboids, finders, HashSet::new()))
}

#[derive(Serialize, Deserialize)]
pub struct State {
    pub finders: Vec<NetFinder>,
    pub yielded_nets: HashSet<Net>,
}

/// Updates the passed `state` with the most recently sent `NetFinder`s, then
/// writes it out to a file.
fn update_and_write_state(
    state: &mut State,
    cuboids: [Cuboid; 2],
    finder_receivers: &mut Vec<Receiver<NetFinder>>,
    channel_rx: &mut Receiver<Receiver<NetFinder>>,
) {
    // Check if there are any neW `NetFinder`s we need to add to our list.
    loop {
        match channel_rx.try_recv() {
            Err(TryRecvError::Empty) => break,
            Ok(rx) => {
                // There should always be an initial state for the `NetFinder` sent
                // immediately after creating the channel.
                let finder = rx.try_recv().unwrap();
                state.finders.push(finder);
                finder_receivers.push(rx);
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
                // Skip over incrementing `i`, since `i` corresponds to the next
                // `NetFinder` in the list now that we're removed one.
                continue;
            }
        }
        i += 1;
    }

    let path = state_path(&cuboids);
    // Initially write to a temporary file so that the previous version is still
    // there if we get Ctrl+C'd while writing or something like that.
    let tmp_path = path.with_extension("json.tmp");
    let file = File::create(&tmp_path).unwrap();
    serde_json::to_writer(BufWriter::new(file), &state).unwrap();
    // Then move it to the real path.
    fs::rename(tmp_path, path).unwrap();
}

pub fn resume(mut cuboids: [Cuboid; 2]) -> anyhow::Result<impl Iterator<Item = Net>> {
    sort_cuboids(&mut cuboids);
    let file = File::open(state_path(&cuboids)).context("no state to resume from")?;
    let state: State = serde_json::from_reader(BufReader::new(file))?;
    Ok(run(cuboids, state.finders, state.yielded_nets))
}

/// A `NetFinder` stored alongside a channel so that it can be sent to another
/// thread which will periodically save our current state.
#[derive(Clone)]
struct ActiveNetFinder {
    finder: NetFinder,
    /// A channel through which copies of `finder` are sent to be saved to a
    /// file recording our current state.
    finder_tx: SyncSender<NetFinder>,
}

impl ActiveNetFinder {
    /// Splits this `ActiveNetFinder` in place, so that its work is now split
    /// between `self` and the returned `ActiveNetFinder`.
    ///
    /// This also returns the receiving end of a channel through which copies of
    /// the current state of the returned `ActiveNetFinder` will be sent.
    ///
    /// Returns `None` if this can't be done, in which case `self` is unchanged.
    fn split(&mut self) -> Option<(ActiveNetFinder, Receiver<NetFinder>)> {
        let mut new_base_index = self.finder.base_index + 1;
        // Find the first instruction that this `NetFinder` controls which has been run,
        // since that's what gets tried first and so that means that the other
        // possibility hasn't been tried yet.
        while self
            .finder
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
        if new_base_index - 1 >= self.finder.queue.len() {
            // We couldn't find one, so this can't be split.
            None
        } else {
            // Create the new `NetFinder` by backtracking this one until the instruction at
            // `new_base_index` hasn't been run.
            let mut new_finder = self.finder.clone();
            while matches!(
                new_finder.queue[new_base_index - 1].state,
                InstructionState::Completed { .. }
            ) {
                new_finder.backtrack();
            }

            self.finder.base_index = new_base_index;
            new_finder.base_index = new_base_index;

            // Now `self.finder` is responsible for the case where the instruction at
            // `new_base_index` is run, and `new_finder` is responsible for the case where
            // it isn't.

            // Create the channel through which we send `NetFinder`s.
            let (finder_tx, finder_rx) = mpsc::sync_channel(1);
            finder_tx.send(new_finder.clone()).unwrap();

            let new_finder = ActiveNetFinder {
                finder: new_finder,
                finder_tx,
            };
            Some((new_finder, finder_rx))
        }
    }
}

/// A `ParallelIterator` over a bunch of `NetFinder`s.
#[derive(Clone)]
struct NetIter {
    finders: Vec<ActiveNetFinder>,
    /// A channel through which the receiving ends of channels for sending the
    /// states of newly created `NetFinder`s are sent.
    channel_tx: Sender<Receiver<NetFinder>>,
    /// A counter used to introduce a delay between when we try to send through
    /// copies of our `NetFinder`s.
    send_counter: u16,
    /// Nets that we're currently in the process of yielding, or
    /// `Finalize::Known(None)` initially.
    yielding: Finalize,
}

impl Iterator for NetIter {
    type Item = Net;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(net) = self.yielding.next() {
                return Some(net);
            }

            // `yielding` either finished or was empty to begin with; find the next
            // potential net and set `yielding` to it before looping back around.

            // When we run out of `NetFinder`s, we're done.
            let Some(ActiveNetFinder { finder, finder_tx }) = self.finders.get_mut(0) else {
                return None;
            };
            while finder.area < finder.target_area && finder.index < finder.queue.len() {
                // Evaluate the next instruction in the queue.
                finder.handle_instruction();

                self.send_counter = self.send_counter.wrapping_add(1);
                if self.send_counter == 0 {
                    let _ = finder_tx.try_send(finder.clone());
                }
            }

            // We broke out of the loop, which means we've reached the end of the queue or
            // the target area. So, finalize the current net to find solutions and set
            // `self.yielding`.
            self.yielding = finder.finalize();

            // Now backtrack and look for more solutions. Note that `self.yielding` is
            // untouched by this.
            if !finder.backtrack() {
                // Backtracking failed which means there are no solutions left, so this
                // `NetFinder` is done.
                self.finders.remove(0);
            }
        }
    }
}

impl Spliterator for NetIter {
    fn split(&mut self) -> Option<Self> {
        let new_finders = if self.finders.len() > 1 {
            // Since `NetFinder`s earlier in the list have more work to do, we split the
            // list up into the first `NetFinder` and everything else.
            Some(self.finders.split_off(1))
        } else if let Some((finder, finder_rx)) =
            self.finders.get_mut(0).and_then(|finder| finder.split())
        {
            self.channel_tx.send(finder_rx).unwrap();
            Some(vec![finder])
        } else {
            None
        };
        new_finders.map(|finders| NetIter {
            finders,
            channel_tx: self.channel_tx.clone(),
            send_counter: 0,
            yielding: Finalize::Known(None),
        })
    }
}

fn run(
    cuboids: [Cuboid; 2],
    finders: Vec<NetFinder>,
    yielded_nets: HashSet<Net>,
) -> impl Iterator<Item = Net> {
    // Create a channel for sending yielded nets to the main thread.
    let (net_tx, net_rx) = mpsc::channel::<Net>();
    // Create channels for sending the states of all our initial `NetFinder`s to the
    // main thread, which will save them to a file.
    let (finder_senders, mut finder_receivers): (Vec<_>, Vec<_>) = finders
        .iter()
        .map(|_| {
            // We specifically limit the buffer to 1 so that threads only send us a new
            // state after we consume the last one.
            mpsc::sync_channel::<NetFinder>(1)
        })
        .unzip();
    // Then create a channel for sending the receiving ends of new such channels
    // that get created down the line.
    let (channel_tx, mut channel_rx) = mpsc::channel();

    let iter = NetIter {
        finders: zip(finders.clone(), finder_senders)
            .map(|(finder, finder_tx)| ActiveNetFinder { finder, finder_tx })
            .collect(),
        channel_tx,
        send_counter: 0,
        yielding: Finalize::Known(None),
    };

    // Spin up the parallel iterator on another thread and send all the results
    // through a channel.
    //
    // Even though rayon uses a thread pool, there unfortunately doesn't seem to be
    // any way to do this without blocking the current thread, which is why we need
    // to spawn a new one.
    thread::Builder::new()
        .name("iter thread".to_owned())
        .spawn(move || {
            iter.par_split()
                .for_each_with(net_tx, |net_tx, net| net_tx.send(net.shrink()).unwrap());
        })
        .unwrap();

    // Create the folder where we're going to store our state.
    fs::create_dir_all(Path::new(env!("CARGO_MANIFEST_DIR")).join("state")).unwrap();

    let mut state = State {
        finders,
        yielded_nets: yielded_nets.clone(),
    };

    // Yield all our previous yielded nets before starting the new ones.
    yielded_nets
        .into_iter()
        .chain(std::iter::from_fn(move || loop {
            match net_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(net) => {
                    let new = state.yielded_nets.insert(net.clone());
                    if new {
                        return Some(net);
                    }
                }
                Err(RecvTimeoutError::Timeout) => update_and_write_state(
                    &mut state,
                    cuboids,
                    &mut finder_receivers,
                    &mut channel_rx,
                ),
                Err(RecvTimeoutError::Disconnected) => {
                    // Update the state with the final value of `yielded_nets`, since it serves as
                    // our way of retrieving results afterwards.
                    update_and_write_state(
                        &mut state,
                        cuboids,
                        &mut finder_receivers,
                        &mut channel_rx,
                    );
                    return None;
                }
            }
        }))
}
