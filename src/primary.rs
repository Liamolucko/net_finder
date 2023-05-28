//! The initial home-grown algorithm I came up with.

use std::{
    collections::HashSet,
    fmt::Write,
    fs::{self, File},
    hash::{Hash, Hasher},
    io::{BufReader, BufWriter},
    iter::zip,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
};

use anyhow::{bail, Context};
use serde::{Deserialize, Serialize};

use crate::{Cuboid, Direction, Face, FacePos, Net, Pos};

use Direction::*;
use Face::*;

#[derive(Serialize, Deserialize, Clone)]
pub struct NetFinder {
    queue: Vec<Instruction>,
    /// The indices of all the 'potential' instructions that we're saving until
    /// the end to run.
    ///
    /// Note that some of these may now be invalid.
    potential: Vec<usize>,

    pub net: Net,
    cuboid_info: [CuboidInfo; 2],
    /// Information about the state of each position in the net.
    pos_states: Vec<PosState>,

    /// The index of the next instruction in `queue` that will be evaluated.
    index: usize,
    /// The size of the net so far.
    pub area: usize,
    pub target_area: usize,
}

/// Various cached info about a particular cuboid.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CuboidInfo {
    /// The cuboid this is info for.
    cuboid: Cuboid,
    /// All the face positions on a cuboid.
    ///
    /// This allows us to represent a face position as an index in this array
    /// instead of having to copy it around everywhere, and also then reuse that
    /// index in a lookup table.
    ///
    /// The lowest 2 bits of an index are always a face position's orientation,
    /// so if you want to check for equality ignoring orientation you can shift
    /// 2 to the right to get rid of them.
    face_positions: Vec<FacePos>,
    /// A lookup table which stores the face position you get by moving in every
    /// possible direction from a face position.
    ///
    /// If `index` is the index of a face position in `face_positions` and
    /// `direction` is the direction you want to find a position adjacent in,
    /// the index you need in this array is `index << 2 | (direction as usize)`.
    adjacent_lookup: Vec<u8>,
    /// A bitfield that encodes whether each square on the cuboid's surface is
    /// filled.
    ///
    /// If `index` is an index of a position in `face_positions`, the index of
    /// the bit repesenting that square (little endian) is `index >> 2`, since
    /// the lower two bits are the orientation which doesn't matter here.
    filled: u64,
}

impl CuboidInfo {
    fn new(cuboid: Cuboid) -> Self {
        let mut face_positions = Vec::new();
        for face in [Bottom, West, North, East, South, Top] {
            for x in 0..cuboid.face_size(face).width {
                for y in 0..cuboid.face_size(face).height {
                    for orientation in 0..4 {
                        face_positions.push(FacePos {
                            face,
                            pos: Pos { x, y },
                            orientation,
                        })
                    }
                }
            }
        }

        let adjacent_lookup = face_positions
            .iter()
            .flat_map(|pos| {
                [Left, Up, Right, Down]
                    .into_iter()
                    .map(|direction| pos.moved_in(direction, cuboid))
            })
            .map(|pos| {
                face_positions
                    .iter()
                    .position(|&other_pos| other_pos == pos)
                    .unwrap()
                    .try_into()
                    .unwrap()
            })
            .collect();

        Self {
            cuboid,
            // This is only a quarter the length of `face_positions` because we don't consider
            // orientation.
            filled: 0,
            face_positions,
            adjacent_lookup,
        }
    }

    /// Given the index of a face position in `self.face_positions`, returns the
    /// index of the face position in `direction` from that position.
    fn adjacent_in(&self, face_pos: u8, direction: Direction) -> u8 {
        self.adjacent_lookup[(usize::from(face_pos) << 2) | direction as usize]
    }

    /// Returns whether the square at a face position is filled.
    fn filled(&self, face_pos: u8) -> bool {
        self.filled & 1 << (face_pos >> 2) != 0
    }

    /// Sets whether the square at a face position is filled.
    fn set_filled(&mut self, face_pos: u8, value: bool) {
        if value {
            self.filled |= 1 << (face_pos >> 2);
        } else {
            self.filled &= !(1 << (face_pos >> 2));
        }
    }

    fn index_of(&self, face_pos: FacePos) -> u8 {
        self.face_positions
            .iter()
            .position(|&pos| pos == face_pos)
            .unwrap()
            .try_into()
            .unwrap()
    }
}

#[test]
fn adjacent_lookup() {
    let cuboid_info = CuboidInfo::new(Cuboid::new(1, 2, 3));
    for (i, face_pos) in cuboid_info.face_positions.iter().enumerate() {
        let i = u8::try_from(i).unwrap();
        for direction in [Left, Up, Right, Down] {
            assert_eq!(
                cuboid_info.face_positions[usize::from(cuboid_info.adjacent_in(i, direction))],
                face_pos.moved_in(direction, cuboid_info.cuboid)
            );
        }
    }
}

/// An instruction to add a square.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Instruction {
    /// The index in `net.squares` where the square will be added.
    net_pos: IndexPos,
    /// The equivalent positions on the surfaces of the cuboids, as indices into
    /// `face_positions` in `CuboidInfo`.
    face_positions: [u8; 2],
    state: InstructionState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum InstructionState {
    /// An instruction which has not been run, either because:
    /// - We haven't gotten to it yet.
    /// - It's invalid.
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
    /// An instruction which has been backtracked.
    ///
    /// An instruction can also have this state if we haven't gotten to it yet
    /// but it was backtracked earlier and we haven't bothered resetting its
    /// state.
    ///
    /// We keep track of this so that we don't waste time considering them as
    /// potential instructions later, since all the cases where it's set will
    /// have already been covered back when it was previously set before being
    /// backtracked.
    Backtracked,
}

impl PartialEq for Instruction {
    fn eq(&self, other: &Self) -> bool {
        self.net_pos == other.net_pos && self.face_positions == other.face_positions
    }
}

impl Eq for Instruction {}

impl Hash for Instruction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.net_pos.hash(state);
        self.face_positions.hash(state);
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
    /// There's an instruction in the queue that wants to set this position to
    /// map to `face_positions`.
    Queued { face_positions: [u8; 2] },
    /// There are two instructions in the queue that want to map this position
    /// to different positions. Any further instructions will see this and not
    /// be added.
    ///
    /// `face_positions` is the what the first instruction wants this net
    /// position to map to, so that this can be restored to `Queued` if/when the
    /// second instruction gets backtracked.
    Invalid { face_positions: [u8; 2] },
    /// This position is filled.
    Filled,
}

impl NetFinder {
    /// Create a bunch of `NetFinder`s that search for all the nets that fold
    /// into all of the passed cuboids.
    fn new(cuboids: &[Cuboid]) -> anyhow::Result<Vec<Self>> {
        let cuboids: [Cuboid; 2] = cuboids.try_into().context("expected 2 cuboids")?;
        if cuboids
            .iter()
            .any(|cuboid| cuboid.surface_area() != cuboids[0].surface_area())
        {
            bail!("all passed cuboids must have the same surface area")
        }

        let net = Net::for_cuboids(&cuboids);
        let middle_x = net.width() / 2;
        let middle_y = net.height() / 2;

        let cuboid_info = cuboids.map(CuboidInfo::new);

        let starting_face_positions: Vec<_> = cuboids[1]
            .surface_squares()
            .into_iter()
            .map(|pos| cuboid_info[1].index_of(pos))
            .map(|index| [0, index])
            .collect();

        Ok(starting_face_positions
            .into_iter()
            .map(|face_positions| {
                // The first instruction is to add the first square.
                let start_pos = IndexPos(
                    usize::from(middle_y) * usize::from(net.width()) + usize::from(middle_x),
                );
                let queue = vec![Instruction {
                    net_pos: start_pos,
                    face_positions,
                    state: InstructionState::NotRun,
                }];
                let mut pos_states = vec![PosState::Untouched; net.squares.len()];
                pos_states[start_pos.0] = PosState::Queued {
                    face_positions: queue[0].face_positions,
                };
                Self {
                    queue,
                    potential: Vec::new(),

                    net: net.clone(),
                    cuboid_info: cuboid_info.clone(),
                    pos_states,

                    index: 0,
                    area: 0,
                    target_area: cuboids[0].surface_area(),
                }
            })
            .collect())
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
            .rfind(|(_, instruction)| matches!(instruction.state, InstructionState::Completed { .. }))
        else {
            return false
        };
        let InstructionState::Completed { followup_index } = instruction.state else { unreachable!() };
        // Mark it as reverted.
        instruction.state = InstructionState::Backtracked;
        self.pos_states[instruction.net_pos.0] = PosState::Queued {
            face_positions: instruction.face_positions,
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
                PosState::Invalid { face_positions } => PosState::Queued { face_positions },
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
        let instruction = &mut self.queue[self.index];
        // Clear `InstructionState::Backtracked` from this instruction if present.
        instruction.state = InstructionState::NotRun;
        if !self.cuboid_info[0].filled(instruction.face_positions[0])
            & !self.cuboid_info[1].filled(instruction.face_positions[1])
            & matches!(
                self.pos_states[instruction.net_pos.0],
                PosState::Queued { .. }
            )
        {
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
                                face_positions: instruction.face_positions,
                            };
                            // We only consider it a real followup instruction if none of the face
                            // positions it sets are already filled.
                            if zip(&self.cuboid_info, instruction.face_positions)
                                .all(|(info, pos)| !info.filled(pos))
                            {
                                followup = true;
                            }
                            self.queue.push(instruction);
                        }
                        // If it's already queued to do the same thing as this instruction, this
                        // instruction is redundant. Don't add it to the queue.
                        PosState::Queued { face_positions }
                            if face_positions == instruction.face_positions => {}
                        // If it's queued to do something different, mark it as invalid and queue
                        // this instruction so that it'll become valid again if/when the instruction
                        // is unqueued.
                        PosState::Queued { face_positions } => {
                            self.pos_states[instruction.net_pos.0] =
                                PosState::Invalid { face_positions };
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
                    self.pos_states[instruction.net_pos.0] = match self.pos_states
                        [instruction.net_pos.0]
                    {
                        PosState::Untouched => unreachable!(),
                        PosState::Queued { .. } => PosState::Untouched,
                        PosState::Invalid { face_positions } => PosState::Queued { face_positions },
                        PosState::Filled => unreachable!(),
                    }
                }
                // Add this to the list of potential instructions.
                self.potential.push(self.index);
            }
        }
        self.index += 1;
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
        let mut face_positions = instruction.face_positions;
        zip(&self.cuboid_info, &mut face_positions)
            .for_each(|(info, pos)| *pos = info.adjacent_in(*pos, direction));
        Some(Instruction {
            net_pos,
            face_positions,
            state: InstructionState::NotRun,
        })
    }

    /// Set a square on the net and surfaces of the cuboids simultaneously,
    /// specifying the square by the index of an instruction which sets it.
    #[inline]
    fn set_square(&mut self, instruction_index: usize, value: bool) {
        let Instruction {
            net_pos,
            ref face_positions,
            ..
        } = self.queue[instruction_index];
        match (self.net.squares[net_pos.0], value) {
            (false, true) => self.area += 1,
            (true, false) => self.area -= 1,
            _ => {}
        }
        self.net.squares[net_pos.0] = value;
        for (info, &pos) in zip(&mut self.cuboid_info, face_positions) {
            info.set_filled(pos, value);
        }
    }

    /// This method, which should be called when the end of the queue is
    /// reached, goes through all of the unrun instructions to find which ones
    /// are valid, and figures out which combinations of them result in a valid
    /// net.
    fn finalize(&self) -> Finalize {
        // First we make sure that there's at least one potential square that sets every
        // square on the surface. We do this before actually constructing the
        // list of potential squares because it's where 99% of calls to this function
        // end, so it needs to be fast.
        // Use a bitfield to store which squares are filled in on each cuboid, starting
        // off with the ones that are already properly filled in.
        let mut filled: [u64; 2] = [0; 2];
        zip(&mut filled, &self.cuboid_info).for_each(|(filled, info)| *filled = info.filled);
        for instruction in self.potential.iter().map(|&index| &self.queue[index]) {
            if !self.cuboid_info[0].filled(instruction.face_positions[0])
                & !self.cuboid_info[1].filled(instruction.face_positions[1])
                & matches!(
                    self.pos_states[instruction.net_pos.0],
                    PosState::Queued { .. }
                )
            {
                for (filled, pos) in zip(&mut filled, instruction.face_positions) {
                    *filled |= 1 << (pos >> 2);
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
            .queue
            .iter()
            .cloned()
            .filter(|instruction| {
                matches!(instruction.state, InstructionState::NotRun)
                    & !self.cuboid_info[0].filled(instruction.face_positions[0])
                    & !self.cuboid_info[1].filled(instruction.face_positions[1])
                    & matches!(
                        self.pos_states[instruction.net_pos.0],
                        PosState::Queued { .. }
                    )
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
            let cuboid_info = &self.cuboid_info[cuboid];
            for square in 0..cuboid_info.face_positions.len() / 4 {
                if !cuboid_info.filled((square << 2).try_into().unwrap()) {
                    // If the square's not already filled, there has to be at least one potential
                    // square that fills it.
                    found.clear();
                    for (i, instruction) in potential_squares.iter().enumerate() {
                        if usize::from(instruction.face_positions[cuboid] >> 2) == square {
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
                        // instruction that sets it, si this is impossible.
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
                            && other_instruction.face_positions != moved_instruction.face_positions
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

pub fn find_nets(cuboids: &[Cuboid]) -> anyhow::Result<impl Iterator<Item = Net>> {
    let finders = NetFinder::new(cuboids)?;

    Ok(run(cuboids, finders, HashSet::new()))
}

#[derive(Serialize, Deserialize)]
pub struct State {
    pub finders: Vec<NetFinder>,
    pub yielded_nets: HashSet<Net>,
}

pub fn resume(cuboids: &[Cuboid]) -> anyhow::Result<impl Iterator<Item = Net>> {
    let file = File::open(state_path(cuboids)).context("no state to resume from")?;
    let state: State = serde_json::from_reader(BufReader::new(file))?;
    Ok(run(cuboids, state.finders, state.yielded_nets))
}

fn run(
    cuboids: &[Cuboid],
    finders: Vec<NetFinder>,
    mut yielded_nets: HashSet<Net>,
) -> impl Iterator<Item = Net> {
    let (net_tx, net_rx) = mpsc::channel();
    let (finder_senders, mut finder_receivers): (Vec<_>, Vec<_>) = finders
        .iter()
        .map(|_| {
            // We specifically limit the buffer to 1 so that threads only send us a new
            // state after we consume the last one.
            mpsc::sync_channel::<NetFinder>(1)
        })
        .unzip();
    for (mut finder, finder_tx) in zip(finders, finder_senders) {
        let net_tx = net_tx.clone();
        thread::spawn(move || {
            // We start from the bottom-left of the bottom face of the cuboid, and work our
            // way out from there. The net is aligned with the bottom face, so left, right,
            // up, down on the net correspond to the same directions on the bottom face.
            // From there they continue up other faces, e.g. left on the net can become up
            // on the left face.
            // We think of directions on each face as they appear from the inside.

            let mut send_counter: u16 = 0;
            loop {
                while finder.area < finder.target_area && finder.index < finder.queue.len() {
                    // Evaluate the next instruction in the queue.
                    finder.handle_instruction();

                    send_counter = send_counter.wrapping_add(1);
                    if send_counter == 0 {
                        let _ = finder_tx.try_send(finder.clone());
                    }
                }

                // We broke out of the loop, which means we've reached the end of the queue or
                // the target area. So, finalize the current net to find solutions and yield
                // them.
                for net in finder.finalize() {
                    if net_tx.send(net.shrink()).is_err() {
                        // The receiver is gone, i.e. the iterator was dropped.
                        return;
                    }
                }

                // Now backtrack and look for more solutions.
                if !finder.backtrack() {
                    // Backtracking failed which means there are no solutions left, we're done.
                    return;
                }
            }
        });
    }

    println!("threads spawned!");

    let (yielded_nets_tx, yielded_nets_rx) = mpsc::channel();

    let cuboids = cuboids.to_vec();
    let yielded_nets_clone = yielded_nets.clone();
    // Spawn a thread whose sole purpose is to periodically record our state to a
    // file.
    thread::spawn(move || {
        // Create the folder where we're going to store our state.
        fs::create_dir_all(Path::new(env!("CARGO_MANIFEST_DIR")).join("state")).unwrap();
        let mut yielded_nets = yielded_nets_clone;
        loop {
            let finders: Vec<_> = finder_receivers
                .iter_mut()
                .filter_map(|rx| rx.recv().ok())
                .collect();

            while let Ok(net) = yielded_nets_rx.try_recv() {
                yielded_nets.insert(net);
            }

            let path = state_path(&cuboids);
            // Initially write to a temporary file so that the previous version is still
            // there if we get Ctrl+C'd while writing or something like that.
            let tmp_path = path.with_extension("json.tmp");
            let file = File::create(&tmp_path).unwrap();
            serde_json::to_writer(
                BufWriter::new(file),
                &State {
                    finders,
                    yielded_nets: yielded_nets.clone(),
                },
            )
            .unwrap();
            // Then move it to the real path.
            fs::rename(tmp_path, path).unwrap();
        }
    });

    // Yield all our previous yielded nets before starting the new ones
    yielded_nets
        .clone()
        .into_iter()
        .chain(net_rx.into_iter().filter(move |net| {
            let new = yielded_nets.insert(net.clone());
            if new {
                yielded_nets_tx.send(net.clone()).unwrap();
            }
            new
        }))
}
