//! The initial home-grown algorithm I came up with.

use std::{
    collections::HashSet,
    fmt::Write,
    fs,
    hash::{Hash, Hasher},
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
    cuboid_info: [CuboidInfo; 2],

    pub net: Net,
    queue: Vec<Instruction>,
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
    /// Whether each square on the cuboid's surface is filled.
    ///
    /// If `index` is an index of a position in `face_positions`, the index in
    /// this array of the square that represents is `index >> 2`, since the
    /// lower two bits are the orientation which doesn't matter here.
    filled: Vec<bool>,
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
            filled: vec![false; face_positions.len() / 4],
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
        self.filled[usize::from(face_pos >> 2)]
    }

    /// Sets whether the square at a face position is filled.
    fn set_filled(&mut self, face_pos: u8, value: bool) {
        self.filled[usize::from(face_pos >> 2)] = value;
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
    /// The instruction has not been run or has been reverted.
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
                    cuboid_info: cuboid_info.clone(),
                    net: net.clone(),
                    queue,
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
        instruction.state = InstructionState::NotRun;
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
        if !self.cuboid_info[0].filled(instruction.face_positions[0])
            & !self.cuboid_info[1].filled(instruction.face_positions[1])
            & matches!(
                self.pos_states[instruction.net_pos.0],
                PosState::Queued { .. }
            )
        {
            // The space is free, so we fill it.
            instruction.state = InstructionState::Completed {
                followup_index: next_index,
            };
            self.pos_states[instruction.net_pos.0] = PosState::Filled;
            let net_pos = instruction.net_pos;
            self.set_square(self.index, true);

            // Add the new things we can do from here to the queue.
            for direction in [Left, Up, Right, Down] {
                if let Some(net_pos) = net_pos.moved_in(direction, &self.net) {
                    // It's easier to just mutate the array in place than turn it into an iterator
                    // and back.
                    let mut face_positions = self.queue[self.index].face_positions;
                    zip(&self.cuboid_info, &mut face_positions)
                        .for_each(|(info, pos)| *pos = info.adjacent_in(*pos, direction));
                    let instruction = Instruction {
                        net_pos,
                        face_positions,
                        state: InstructionState::NotRun,
                    };

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
        }
        self.index += 1;
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
}

fn state_path(cuboids: &[Cuboid]) -> PathBuf {
    let mut name = "state/".to_string();
    for (i, cuboid) in cuboids.iter().enumerate() {
        if i != 0 {
            name.push(',');
        }
        write!(name, "{cuboid}").unwrap();
    }
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
    let bytes = fs::read(state_path(cuboids)).context("no state to resume from")?;
    let state: State = postcard::from_bytes(&bytes)?;
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
                while finder.area < finder.target_area {
                    while finder.index >= finder.queue.len() {
                        // We've exhausted the queue and haven't found a solution, so we must have
                        // done something wrong. Backtrack.
                        if !finder.backtrack() {
                            // We're out of options, so we're done.
                            return;
                        }
                    }

                    // Evaluate the next instruction in the queue.
                    finder.handle_instruction();

                    send_counter = send_counter.wrapping_add(1);
                    if send_counter == 0 {
                        let _ = finder_tx.try_send(finder.clone());
                    }
                }

                // We broke out of the loop, which means we reached the target area and have a
                // valid net!
                let net = finder.net.shrink();
                finder.backtrack();

                if net_tx.send(net).is_err() {
                    // The receiver is gone, i.e. the iterator was dropped.
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

            fs::write(
                state_path(&cuboids),
                &postcard::to_stdvec(&State {
                    finders,
                    yielded_nets: yielded_nets.clone(),
                })
                .unwrap(),
            )
            .unwrap();
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
