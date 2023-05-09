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
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{Cuboid, Direction::*, FacePos, Net, Pos, Surface};

#[derive(Serialize, Deserialize, Clone)]
pub struct NetFinder {
    pub cuboids: Vec<Cuboid>,
    pub net: Net,
    surfaces: Vec<Surface>,
    queue: Vec<Instruction>,
    /// A map from net positions to the possible positions they could map to on
    /// the cuboids' surfaces, annotated with how many times they pop up in the
    /// queue.
    ///
    /// There are only at most 4 - one from each adjacent position.
    pub pos_possibilities: FxHashMap<Pos, heapless::Vec<heapless::Vec<FacePos, 3>, 4>>,
    /// The index of the next instruction in `queue` that will be evaluated.
    index: usize,
    /// The size of the net so far.
    pub area: usize,
    pub target_area: usize,
}

/// An instruction to add a square.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Instruction {
    /// The position on the net where the square will be added.
    net_pos: Pos,
    /// The equivalent positions on the surfaces of the cuboids.
    ///
    /// This isn't actually supposed to represent an array of length 2, it's an
    /// array of length *up to* 2; later elements are just ignored since it's
    /// always zipped with something or other.
    face_positions: heapless::Vec<FacePos, 3>,
    state: InstructionState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum InstructionState {
    /// The instruction has not been run or has been reverted.
    NotRun,
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

impl NetFinder {
    /// Create a bunch of `NetFinder`s that search for all the nets that fold
    /// into all of the passed cuboids.
    fn new(cuboids: &[Cuboid]) -> anyhow::Result<Vec<Self>> {
        if cuboids.is_empty() {
            bail!("at least one cuboid must be provided")
        }
        if cuboids
            .iter()
            .any(|cuboid| cuboid.surface_area() != cuboids[0].surface_area())
        {
            bail!("all passed cuboids must have the same surface area")
        }

        let net = Net::for_cuboids(cuboids);
        let middle_x = net.width() / 2;
        let middle_y = net.height() / 2;

        // The state of how much each face of the cuboid has been filled in. I reused
        // `Net` to represent the state of each face.
        // I put them in the arbitrary order of bottom, west, north, east, south, top.
        let surfaces: Vec<_> = cuboids.iter().copied().map(Surface::new).collect();

        // All the starting points on each cuboid we want to consider.
        let mut starting_points: Vec<Vec<FacePos>> = cuboids
            .iter()
            .map(|cuboid| cuboid.surface_squares())
            .collect();
        // We only need to consider one spot on one of them. The only reason we're
        // considering multiple starting spots is so that the same spot on the net can
        // map to any different spot on the different cuboids; for a single cuboid, we
        // can discover a net from any starting spot.
        starting_points[0] = vec![FacePos::new()];

        let mut starting_face_positions: Vec<heapless::Vec<FacePos, 3>> = Vec::new();
        let mut indices: Vec<_> = cuboids.iter().map(|_| 0_usize).collect();
        'outer: loop {
            starting_face_positions.push(
                zip(&indices, &starting_points)
                    .map(|(&index, vec)| vec[index])
                    .collect(),
            );
            for index_index in (0..indices.len()).rev() {
                indices[index_index] += 1;
                if indices[index_index] >= starting_points[index_index].len() {
                    indices[index_index] = 0;
                    if index_index == 0 {
                        break 'outer;
                    }
                } else {
                    break;
                }
            }
        }

        Ok(starting_face_positions
            .into_iter()
            .map(|face_positions| {
                // The first instruction is to add the first square.
                let queue = vec![Instruction {
                    net_pos: Pos::new(middle_x, middle_y),
                    face_positions,
                    state: InstructionState::NotRun,
                }];
                let mut pos_possibilities = FxHashMap::default();
                pos_possibilities.insert(
                    queue[0].net_pos,
                    [queue[0].face_positions.clone()]
                        .as_slice()
                        .try_into()
                        .unwrap(),
                );
                Self {
                    cuboids: cuboids.to_owned(),
                    net: net.clone(),
                    surfaces: surfaces.clone(),
                    queue,
                    pos_possibilities,
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
            .rfind(|(_, instruction)| matches!(instruction.state, InstructionState::Completed { .. })) else {
                return false
            };
        let InstructionState::Completed { followup_index } = instruction.state else { unreachable!() };
        // Mark it as reverted.
        instruction.state = InstructionState::NotRun;
        // Remove the square it added.
        self.set_square(last_success_index, false);
        // Then remove all the instructions added as a result of this square.
        for instruction in self.queue.drain(followup_index..) {
            self.pos_possibilities
                .get_mut(&instruction.net_pos)
                .unwrap()
                .retain(|face_positions| *face_positions != instruction.face_positions);
        }
        // We continue executing from just after the instruction we undid.
        self.index = last_success_index + 1;
        true
    }

    /// Handle the instruction at the current index in the queue, incrementing
    /// the index afterwards.
    #[inline]
    fn handle_instruction(&mut self) {
        // for (index, instruction) in self.queue.iter().enumerate() {
        //     println!(
        //         "{marker}{index}: {}: {:?}",
        //         instruction.net_pos,
        //         instruction.state,
        //         marker = if index == self.index { '>' } else { ' ' }
        //     );
        // }
        // println!("----");
        let next_index = self.queue.len();
        let instruction = &mut self.queue[self.index];
        if !zip(&self.surfaces, &instruction.face_positions)
            .any(|(surface, &pos)| surface.filled(pos))
            && self.pos_possibilities[&instruction.net_pos].len() == 1
        {
            // The space is free, so we fill it.
            instruction.state = InstructionState::Completed {
                followup_index: next_index,
            };
            let net_pos = instruction.net_pos;
            self.set_square(self.index, true);

            // Add the new things we can do from here to the queue.
            for direction in [Left, Up, Right, Down] {
                if let Some(net_pos) = net_pos.moved_in(direction, self.net.size()) {
                    // `heapless::Vec`'s `FromIterator` impl isn't very efficient, which is why we
                    // clone and modify instead.
                    let mut face_positions = self.queue[self.index].face_positions.clone();
                    zip(&self.cuboids, &mut face_positions)
                        .for_each(|(&cuboid, pos)| pos.move_in(direction, cuboid));
                    let instruction = Instruction {
                        net_pos,
                        face_positions,
                        state: InstructionState::NotRun,
                    };
                    let pos_possibilities = self
                        .pos_possibilities
                        .entry(instruction.net_pos)
                        .or_insert(heapless::Vec::new());
                    if !pos_possibilities.contains(&instruction.face_positions) {
                        // Note: if the length of `pos_possibilities` here isn't 0, this instruction
                        // is invalid and will never be carried out.
                        // However, we enqueue it anyway, so that when we get to the existing
                        // instruction that wants to put a square at the same spot
                        // on the net which corresponds to a different spot on the
                        // surface, it'll see our addition to `pos_possibilities` and not
                        // run.
                        pos_possibilities
                            .push(instruction.face_positions.clone())
                            .unwrap();
                        self.queue.push(instruction);
                    }
                }
            }
        }
        self.index += 1;
    }

    /// Set a square on the net and surfaces of the cuboids simultaneously,
    /// specifying the square by the index of an instruction which sets it.
    fn set_square(&mut self, instruction_index: usize, value: bool) {
        let Instruction {
            net_pos,
            ref face_positions,
            ..
        } = self.queue[instruction_index];
        match (self.net.filled(net_pos), value) {
            (false, true) => self.area += 1,
            (true, false) => self.area -= 1,
            _ => {}
        }
        self.net.set(net_pos, value);
        for (surface, &pos) in zip(&mut self.surfaces, face_positions) {
            surface.set(pos, value);
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
