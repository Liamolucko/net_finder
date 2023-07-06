use std::{
    array,
    cell::Cell,
    collections::VecDeque,
    iter::zip,
    mem,
    sync::mpsc,
    thread::{self, ScopedJoinHandle},
    time::{Duration, Instant},
};

use indicatif::{HumanBytes, MultiProgress, ProgressBar, ProgressStyle};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{Cuboid, Direction, Face, FacePos, Pos};

use super::{Node, NodeRef, Zdd256};

use Direction::*;
use Face::*;

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
}

impl CuboidInfo {
    fn new(cuboid: Cuboid) -> Self {
        let mut face_positions = Vec::new();
        // Create the list of face positions by starting with one spot and spiralling out from there, to try and make all the squares exit as quickly as possible.
        for orientation in 0..4 {
            face_positions.push(FacePos {
                face: Bottom,
                pos: Pos::new(0, 0),
                orientation,
            });
        }
        let mut index = 0;
        while index < face_positions.len() {
            for direction in [Left, Down, Up, Right] {
                let pos = face_positions[index].moved_in(direction, cuboid);
                if !face_positions.contains(&pos) {
                    for orientation in 0..4 {
                        face_positions.push(FacePos {
                            face: pos.face,
                            pos: pos.pos,
                            orientation,
                        });
                    }
                }
            }
            index += 4;
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
            face_positions,
            adjacent_lookup,
        }
    }

    /// Given the index of a face position in `self.face_positions`, returns the
    /// index of the face position in `direction` from that position.
    fn adjacent_in(&self, face_pos: u8, direction: Direction) -> u8 {
        self.adjacent_lookup[(usize::from(face_pos) << 2) | direction as usize]
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

/// Statically known information about a pair of cuboids.
struct StaticInfo {
    /// Information about the cuboids.
    cuboid_info: [CuboidInfo; 2],
}

impl StaticInfo {
    fn for_cuboids(cuboids: [Cuboid; 2]) -> Self {
        Self {
            cuboid_info: cuboids.map(CuboidInfo::new),
        }
    }

    /// Returns the mapping that would need to be present in `direction` from
    /// `mapping` for `mapping` to connect to it.
    fn adjacent_mapping(&self, mapping: (u8, u8), direction: Direction) -> (u8, u8) {
        let (square, mapping) = mapping;
        let (moved_square, square_orientation) =
            split_face_pos(self.cuboid_info[0].adjacent_in(square << 2, direction));
        let (moved_mapping_square, mapping_orientation) =
            split_face_pos(self.cuboid_info[0].adjacent_in(mapping, direction));
        // Adjust the orientation of the position that `square` maps to to be relative
        // to `square`.
        let moved_mapping = face_pos(
            moved_mapping_square,
            mapping_orientation - square_orientation,
        );
        (moved_square, moved_mapping)
    }
}

/// The minimum amount of information about a partially-mapped pair of cuboids
/// necessary to determine which combinations of future mappings are valid.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NodeState {
    /// A bitmask of which squares on the second cuboid are still valid to map
    /// to, ignoring direction.
    valid: u64,
    /// The mappings of all the squares that are in the frontier.
    mappings: [u8; 64],
    /// The connected components that all the squares in the frontier are in.
    components: [u8; 64],
}

impl NodeState {
    /// Calculates the edges coming out of a node with this state, which
    /// represent choosing the mapping for `square`.
    ///
    /// `frontier` is the list of squares which have entered and haven't exited
    /// yet at the _start_ of this function call, including the new square that
    /// is being mapped by these edges.
    ///
    /// `exiting` is the number of squares at the start of `frontier` which are
    /// going to exit at the end of this function call.
    fn edges(
        &self,
        square: u8,
        info: &StaticInfo,
        frontier: &[u8],
        exiting: u8,
    ) -> [Result<NodeState, ConstantNode>; 256] {
        array::from_fn(|mapping| {
            let mapping: u8 = mapping.try_into().unwrap();
            if self.valid & (1 << (mapping / 4)) == 0 {
                return Err(ConstantNode::Zero);
            }

            let mut new_state = self.clone();
            // Apply the mapping of this square.
            new_state.mappings[usize::from(square)] = mapping;
            new_state.valid &= !(1 << (mapping / 4));
            // Create a new connected component that just contains this square.
            new_state.components[usize::from(square)] = square;

            for direction in [Left, Up, Right, Down] {
                let (parent, parent_mapping) = info.adjacent_mapping((square, mapping), direction);
                if parent > square {
                    continue;
                }
                if new_state.mappings[usize::from(parent)] == parent_mapping {
                    // Merge together `square` and `parent`'s connected
                    // components.
                    // Note that `square`'s connected component doesn't
                    // necessarily still just contain itself, since it might
                    // have merged with another parent's connected component in
                    // an earlier iteration of this loop.
                    let index1 = new_state.components[usize::from(square)];
                    let index2 = new_state.components[usize::from(parent)];
                    let old_index = u8::max(index1, index2);
                    let new_index = u8::min(index1, index2);
                    new_state.update_component_index(old_index, new_index, frontier)
                }
            }

            // Handle exiting squares.
            // We update the frontier as squares exit.
            let mut frontier = frontier;
            for _ in 0..exiting {
                let (&square, new_frontier) = frontier.split_first().unwrap();
                frontier = new_frontier;
                new_state.handle_square_exit(square, frontier)?;
            }

            Ok(new_state)
        })
    }

    /// Changes the component index of everything with component index
    /// `old_index` to `new_index`.
    fn update_component_index(&mut self, old_index: u8, new_index: u8, frontier: &[u8]) {
        for square in frontier.iter().copied() {
            if self.components[usize::from(square)] == old_index {
                self.components[usize::from(square)] = new_index;
            }
        }
    }

    /// Handles a square exiting.
    fn handle_square_exit(&mut self, square: u8, frontier: &[u8]) -> Result<(), ConstantNode> {
        if self.components[usize::from(square)] == square {
            // We need to update the index of the square's component to the index of the
            // square with the next smallest index.
            let old_index = self.components[usize::from(square)];
            let new_index = frontier
                .iter()
                .copied()
                .find(|&square| self.components[usize::from(square)] == old_index);
            if let Some(new_index) = new_index {
                self.update_component_index(old_index, new_index, frontier);
            } else {
                // This is the last square in this component, so the component is exiting too!
                // This means that either we're done or this is invalid - if this was the last
                // square, we've successfully made a connected component that covers everything,
                // but if not, that means we've made a disconnected component which isn't
                // allowed.
                if frontier.is_empty() {
                    return Err(ConstantNode::One);
                } else {
                    return Err(ConstantNode::Zero);
                }
            }
        }

        // Delete all the data associated with this square.
        self.components[usize::from(square)] = 0;
        self.mappings[usize::from(square)] = 0;

        Ok(())
    }

    fn new(squares: u8) -> NodeState {
        NodeState {
            valid: 1u64.wrapping_shl(squares.into()).wrapping_sub(1),
            mappings: [0; 64],
            components: [0; 64],
        }
    }
}

/// One of the constant nodes (the 0-node or the 1-node). Used as an `Err`
/// variant for when an edge should point to one of these instead of a new node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ConstantNode {
    /// The 0-node.
    Zero,
    /// The 1-node.
    One,
}

/// Builds a face position from a square and an orientation.
fn face_pos(square: u8, orientation: i8) -> u8 {
    (square << 2) | (orientation & 0b11) as u8
}

/// Splits a face position up into a square and an orientation.
fn split_face_pos(face_pos: u8) -> (u8, i8) {
    (face_pos >> 2, (face_pos & 0b11) as i8)
}

fn row_thread(
    info: &StaticInfo,
    frontier: &[u8],
    exiting: u8,
    progress_bars: &[ProgressBar],
    index: usize,
    receiver: mpsc::Receiver<NodeState>,
    sender: Option<mpsc::Sender<NodeState>>,
) -> Vec<Node> {
    let progress_bar = progress_bars[index].clone();
    let next_progress_bar = progress_bars.get(index + 1).cloned();

    // The row of nodes that this thread is building up.
    let mut row: Vec<Node> = Vec::new();

    // Store the the `NodeState`s we're passing off to the next row for
    // de-duplication.
    let mut yielded_states: FxHashMap<NodeState, u32> = FxHashMap::default();
    // A hack to work around not being able to access `yielded_states.len()` since
    // it's exclusively borrowed by `handle_state`.
    let num_yielded_states = Cell::new(0);

    // Create a helper function that takes the result of `NodeState::zero_edge` or
    // `NodeState::one_edge` and turns it into a `NodeRef`, handling all the
    // de-duping and such.
    let mut handle_state = |result: Result<NodeState, ConstantNode>| -> NodeRef {
        match result {
            Ok(new_node) => {
                if let Some(&index) = yielded_states.get(&new_node) {
                    NodeRef::NextRow {
                        index: index.try_into().unwrap(),
                    }
                } else {
                    // We've found a new `NodeState`, send it to the next row.
                    sender
                        .as_ref()
                        .expect("node in last row doesn't lead to a constant node")
                        .send(new_node.clone())
                        .unwrap();
                    let index = yielded_states.len();
                    yielded_states.insert(
                        new_node,
                        index.try_into().expect("more than u32::MAX nodes in a row"),
                    );
                    num_yielded_states.set(yielded_states.len());
                    NodeRef::NextRow { index }
                }
            }
            Err(constant_node) => match constant_node {
                ConstantNode::Zero => NodeRef::Zero,
                ConstantNode::One => NodeRef::One,
            },
        }
    };

    let mut progress_last_updated = Instant::now();

    // Receive `NodeState`s from the previous row and turn them into actual `Node`s.
    for state in receiver {
        // The index of the row is the square that edges coming out of it are choosing the mapping for.
        let edges = state
            .edges(index.try_into().unwrap(), info, frontier, exiting)
            .map(&mut handle_state);

        // Now add the new node to the row.
        row.push(Node::new(edges));

        // Updating the progress bar is surprisingly slow thanks to some mutex stuff, so
        // only do it every 50ms.
        if progress_last_updated.elapsed() > Duration::from_millis(50) {
            progress_bar.set_position(row.len().try_into().unwrap());
            progress_bar.set_message(format!(
                "mem. usage: nodes: {}, states: {}, queue: {}",
                HumanBytes((row.len() * mem::size_of::<Node>()) as u64),
                HumanBytes((num_yielded_states.get() * mem::size_of::<NodeState>()) as u64),
                HumanBytes(
                    progress_bar
                        .length()
                        .unwrap()
                        .saturating_sub(row.len() as u64)
                        * mem::size_of::<NodeState>() as u64
                )
            ));
            // Set the length of the next bar to the number of `NodeState`s we've sent it to
            // process.
            if let Some(next_progress_bar) = &next_progress_bar {
                next_progress_bar.set_length(num_yielded_states.get() as u64);
            }
            progress_last_updated = Instant::now();
        }
    }

    progress_bar.set_position(row.len().try_into().unwrap());
    progress_bar.set_length(row.len().try_into().unwrap());
    progress_bar.finish_with_message("done");
    if let Some(next_progress_bar) = &next_progress_bar {
        next_progress_bar.set_length(num_yielded_states.get() as u64);
    }

    row
}

impl Zdd256 {
    pub fn construct(cuboids: [Cuboid; 2]) -> Self {
        assert_eq!(cuboids[0].surface_area(), cuboids[1].surface_area());
        let squares: u8 = cuboids[0].surface_area().try_into().unwrap();
        assert!(squares <= 64);
        let info = StaticInfo::for_cuboids(cuboids);

        // Create the list of 'frontiers', which is basically the list of vertices which
        // are still 'relevant' at a given row of the ZDD.
        //
        // The frontier at a given point in time is all of the squares which have
        // 'entered' and have not yet 'exited'.
        //
        // A square enters when we choose its mapping, and exits when its last
        // neighbour's mapping is chosen (or immediately after entering, if it comes
        // after all of its neighbours in the list).
        //
        // We create the frontier at the _start_ of each row: since vertices exit in the
        // process of moving from that row to the next row, the frontier is different at
        // the start and the end.
        //
        // We do this in a bit of a tactical fashion, where all of the squares that are
        // going to exit are placed at the start of the frontier, so that as we're
        // dealing with squares exiting, we can just slice off one element from the
        // start of the frontier to get the current, intermediate version of the
        // frontier.
        //
        // Each frontier is stored alongside the number of about-to-exit squares there
        // are at the start of said frontier.
        let mut frontiers: Vec<(u8, Vec<u8>)> = vec![(0, vec![]); usize::from(squares)];
        for square in 0..squares {
            let entry = square;
            let exit = [Left, Up, Right, Down]
                .into_iter()
                .map(|direction| info.cuboid_info[0].adjacent_in(square << 2, direction) >> 2)
                .chain([square])
                .max()
                .unwrap();
            // Add this square to the end of all the frontiers where it's there the whole
            // time.
            for index in entry..exit {
                frontiers[usize::from(index)].1.push(square);
            }
            // Then add it to the start of the frontier where it exits halfway through.
            let (exiting, frontier) = &mut frontiers[usize::from(exit)];
            frontier.insert(0, square);
            *exiting += 1;
        }

        for (exiting, frontier) in &frontiers {
            println!("{}, {}", exiting, frontier.len());
        }

        // Create a progress bar for each thread, which is initially hidden until the
        // thread gets its first `NodeState` to process.
        let progress = MultiProgress::new();
        let style = ProgressStyle::with_template(
            "{prefix} - {msg} {wide_bar} {human_pos} / {human_len} nodes",
        )
        .unwrap();
        let progress_bars: Vec<_> = (0..squares)
            .map(|i| {
                let bar = ProgressBar::new_spinner()
                    .with_prefix(format!("row {}", i + 1))
                    .with_style(style.clone());
                bar.set_length(0);
                progress.add(bar.clone());
                bar
            })
            .collect();

        let rows = thread::scope(|s| {
            let (senders, receivers): (Vec<_>, Vec<_>) =
                (0..squares).map(|_| mpsc::channel::<NodeState>()).unzip();
            // Extract the first sender, which is used to send the initial `NodeState` to
            // the thread handling the first row.
            let mut senders = senders.into_iter();
            let first_sender = senders.next().unwrap();
            first_sender.send(NodeState::new(squares)).unwrap();
            drop(first_sender);

            let mut handles: VecDeque<ScopedJoinHandle<_>> = VecDeque::new();
            let mut rows = Vec::new();
            for (index, (sender, receiver)) in
                zip(senders.map(Some).chain([None]), receivers).enumerate()
            {
                if handles.len() >= thread::available_parallelism().unwrap().into() {
                    let row = handles.pop_front().unwrap().join().unwrap();
                    rows.push(row);
                }
                let info = &info;
                let (exiting, ref frontier) = frontiers[index];
                let progress_bars = &progress_bars;
                let handle = thread::Builder::new()
                    .name(format!("row {}", index + 1))
                    .spawn_scoped(s, move || {
                        row_thread(
                            info,
                            frontier,
                            exiting,
                            progress_bars,
                            index,
                            receiver,
                            sender,
                        )
                    })
                    .unwrap();
                handles.push_back(handle);
            }

            rows.extend(handles.into_iter().map(|handle| handle.join().unwrap()));
            rows
        });

        Self { rows }
    }
}
