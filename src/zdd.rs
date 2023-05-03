// This implements the ZDD-based algorithm for finding common developments of cuboids described in http://dx.doi.org/10.1016/j.comgeo.2017.03.001.

use std::{
    cell::Cell,
    collections::{HashSet, VecDeque},
    hash::Hash,
    iter::zip,
    mem,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use crate::{Cuboid, Direction, Face, FacePos, Net, Pos};

use bitvec::prelude::*;
use indicatif::{HumanBytes, MultiProgress, ProgressBar, ProgressStyle};

use rayon::{
    iter::plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer},
    prelude::ParallelIterator,
};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use Direction::*;
use Face::*;

/// A zero-suppressed binary decision diagram (ZDD), specialised for the purpose
/// of storing developments of a cuboid.
///
/// There are some caveats with this; it doesn't produce output quite as clean
/// as `NetFinder`:
/// * It doesn't de-duplicate rotations and such of the same net.
/// * It doesn't check to make sure that no cuts are required.
///
/// So, the results have to be filtered a bit after the fact to account for
/// that.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Zdd {
    /// The cuboid this is a ZDD for.
    cuboid: Cuboid,
    /// The vertices of the cuboid this is a ZDD for.
    ///
    /// The first 8 vertices are the corners of the cuboid, so that we can
    /// easily tell which vertices are corners by just looking at the index.
    vertices: Vec<Vertex>,
    /// The edges of the cuboid this is a ZDD for.
    edges: Vec<Edge>,

    /// The rows of nodes in the ZDD.
    rows: Vec<Vec<Node>>,
}

/// A node in a ZDD.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Node {
    /// The combination of constant nodes and references to the next row that this node contains.
    state: u8,
    /// If the 0-edge of this node points to a node in the next row, the index of that node.
    zero_edge_index: u32,
    /// If the 1-edge of this node points to a node in the next row, the index of that node.
    one_edge_index: u32,
}

impl Node {
    fn new(zero_edge: NodeRef, one_edge: NodeRef) -> Self {
        let mut state = 0;
        let mut zero_edge_index = 0;
        let mut one_edge_index = 0;

        match zero_edge {
            NodeRef::Zero => state |= 0b0000,
            NodeRef::One => state |= 0b0001,
            NodeRef::NextRow { index } => {
                state |= 0b0010;
                zero_edge_index = index.try_into().expect("more than u32::MAX nodes in a row");
            }
        }
        match one_edge {
            NodeRef::Zero => state |= 0b0000,
            NodeRef::One => state |= 0b0100,
            NodeRef::NextRow { index } => {
                state |= 0b1000;
                one_edge_index = index.try_into().expect("more than u32::MAX nodes in a row");
            }
        }

        Self {
            state,
            zero_edge_index,
            one_edge_index,
        }
    }

    fn zero_edge(&self) -> NodeRef {
        match self.state & 0b0011 {
            0b0000 => NodeRef::Zero,
            0b0001 => NodeRef::One,
            0b0010 => NodeRef::NextRow {
                index: self.zero_edge_index.try_into().unwrap(),
            },
            _ => unreachable!(),
        }
    }

    fn one_edge(&self) -> NodeRef {
        match self.state & 0b1100 {
            0b0000 => NodeRef::Zero,
            0b0100 => NodeRef::One,
            0b1000 => NodeRef::NextRow {
                index: self.one_edge_index.try_into().unwrap(),
            },
            _ => unreachable!(),
        }
    }
}

/// The state of the graph represented by a node in a ZDD.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NodeState {
    /// The indices of the connected components that each vertex in the graph is
    /// in.
    components: Vec<u8>,
    /// Whether each vertex in the graph is a pendant vertex.
    ///
    /// The original algorithm stores the degree of each vertex, but the only
    /// thing that actually matters is whether the degree is 0, 1, or > 1.
    /// You can already tell whether it's zero by checking `sizes`, and so all
    /// we need is this set of booleans for whether the degree is 1.
    pendant: BitVec,
}

impl NodeState {
    /// Create a new node without any info filled in.
    fn new(num_vertices: u8) -> Self {
        Self {
            components: (0..num_vertices).collect(),
            pendant: bitvec![0; num_vertices.into()],
        }
    }

    fn size(&self, vertex: usize) -> usize {
        let component = self.components[vertex];
        self.components
            .iter()
            .filter(|&&other_component| other_component == component)
            .count()
    }

    /// This function should be called when a vertex is part of an edge for the
    /// last time (is leaving the frontier), and checks that it satisfies all
    /// our requirements now that it's never going to be touched again.
    //
    /// If those requirements aren't met, it returns `Err(ConstantNode::Zero)`,
    /// which effectively discards this possiblility.
    fn handle_vertex_exit(&self, vertex: usize) -> Result<(), ConstantNode> {
        if (0..8).contains(&vertex) {
            // The first 8 vertices are the corners of the cuboid.
            // We require that corners have at least one edge coming out of them; in other
            // words, that they're part of a connected component with size > 1.
            if self.size(vertex) < 2 {
                return Err(ConstantNode::Zero);
            }
        } else {
            // Vertices that aren't corners are required to not be pendant vertices, since
            // they should just be glued together if that's the case.
            if self.pendant[vertex] {
                return Err(ConstantNode::Zero);
            }
        }
        Ok(())
    }

    /// When a vertex is exiting, checks whether a vertex is the last vertex in
    /// its connected component to exit, and if it is, handles the 'exit' of the
    /// component.
    ///
    /// What this actually means is that it checks whether the component
    /// contains more than just the one vertex, and if it does, that means that
    /// the graph is now either invalid or complete.
    ///
    /// Since the graph is only allowed to have at most one connected component
    /// with size > 1, if one such component is never going to be touched again
    /// (i.e., will never get merged with the other ones), it'd better be the
    /// only one.
    ///
    /// So, we first check whether there are any other connected components with
    /// size > 1, and fail if so. Otherwise, the graph is done, and we check the
    /// degree constraints for any remaining vertices before returning
    /// `Err(ConstantNode::One)`.
    fn handle_component_exit(
        &self,
        vertex: usize,
        remaining_edges: &[Edge],
        frontier: &[usize],
    ) -> Result<(), ConstantNode> {
        if !frontier
            .iter()
            .any(|&other_vertex| self.components[other_vertex] == self.components[vertex])
        {
            // If there aren't any vertices left in the frontier that are in the same
            // connected component as this vertex, this component is now also never going to
            // be modified again.

            // If the component is of size > 1, it's _the_ connected component of the graph,
            // since there's only allowed to be one with size > 1.
            if self.size(vertex) > 1 {
                if frontier.iter().any(|&vertex| self.size(vertex) > 1) {
                    // There isn't allowed to be more than one connected component with size > 1, so
                    // this case is invalid.
                    return Err(ConstantNode::Zero);
                }

                // The graph is now finished, since there aren't any more edges that would
                // expand this connected component, and creating any new connected components
                // with size > 1 is illegal.
                //
                // Before we return `ConstantNode::One` to signal that, we handle the exit of
                // all the vertices that haven't already exited, since we're cutting things off
                // early and those checks won't get performed when they normally would when the
                // vertices exit the frontier.
                //
                // Note that we can't just use the frontier for this because it's possible that
                // there are some vertices that still haven't yet entered the frontier, and we
                // need to check those too.
                let remaining_vertices: HashSet<usize> = remaining_edges
                    .iter()
                    .flat_map(|edge| edge.vertices)
                    .collect();
                for vertex in remaining_vertices {
                    self.handle_vertex_exit(vertex)?;
                }

                // Now we finish the graph.
                return Err(ConstantNode::One);
            }
        }

        Ok(())
    }

    /// Returns the node that the 0-edge of this node should point to, when
    /// `edge` is being added.
    fn zero_edge(
        &self,
        edge: Edge,
        remaining_edges: &[Edge],
        frontier: &[usize],
    ) -> Result<NodeState, ConstantNode> {
        // First handle the exits of the individual vertices.
        for vertex in edge.vertices {
            if !frontier.contains(&vertex) {
                self.handle_vertex_exit(vertex)?;
            }
        }

        // Create a modified version of the frontier which still contains the vertices
        // of this edge, so that the first vertex still sees the second vertex as not
        // having exited yet as it's exiting.
        //
        // The point of this is to make sure that if both vertices are exiting, and
        // causing their components to exit, at the same time, the first vertex to exit
        // sees the other one, and can make sure we don't have two connected components
        // of size > 1.
        //
        // We only need to do this for the 0-edge because in the 1-edge, both vertices
        // are in the same component anyway.
        let mut modified_frontier = frontier.to_vec();
        for vertex in edge.vertices {
            if !modified_frontier.contains(&vertex) {
                modified_frontier.push(vertex);
            }
        }

        // Then check if the exits of these vertices result in the exit of their
        // connected components.
        for vertex in edge.vertices {
            if !frontier.contains(&vertex) {
                modified_frontier.swap_remove(
                    modified_frontier
                        .iter()
                        .rposition(|v| *v == vertex)
                        .unwrap(),
                );

                // This is the last edge containing the node, so we now check that all the
                // conditions we place on nodes are satisfied.
                self.handle_component_exit(vertex, remaining_edges, &modified_frontier)?
            }
        }

        Ok(self.clone())
    }

    /// Returns the node that the 1-edge of this node should point to, when
    /// `edge` is being added.
    fn one_edge(
        &self,
        edge: Edge,
        remaining_edges: &[Edge],
        frontier: &[usize],
    ) -> Result<NodeState, ConstantNode> {
        if self.components[edge.vertices[0]] == self.components[edge.vertices[1]] {
            // Adding this edge would create a cycle, which isn't allowed, so
            // discard this possibility by pointing to the zero-edge.
            return Err(ConstantNode::Zero);
        }

        let mut new_node = self.clone();

        // Actually update the info in `new_node` to account for this edge being added.
        // First we update whether or not the vertices are pendant.
        for vertex in edge.vertices {
            if new_node.size(vertex) == 1 {
                // This vertex is getting its first edge added to it, and becoming pendant.
                new_node.pendant.set(vertex, true);
            } else if new_node.pendant[vertex] {
                // This pendant vertex is getting a second edge added to it, and is no longer
                // pendant.
                new_node.pendant.set(vertex, false);
            }
        }

        // Then we merge the two components that the endpoints of the edge are in.
        let comp1 = new_node.components[edge.vertices[0]];
        let comp2 = new_node.components[edge.vertices[1]];
        let min_comp = u8::min(comp1, comp2);
        let max_comp = u8::max(comp1, comp2);

        // The minimum of the two component indices is the new index for the merged
        // component.
        let new_index = min_comp;

        for &vertex in frontier.iter().chain(&edge.vertices) {
            // Set the component index for any vertices with the old component index to the
            // new index.
            if new_node.components[vertex] == max_comp {
                new_node.components[vertex] = new_index;
            }
        }

        // Then we handle exiting vertices.
        for vertex in edge.vertices {
            if !frontier.contains(&vertex) {
                self.handle_vertex_exit(vertex)?;
            }
        }

        // Then, if both vertices are exiting, we check if the component we've just
        // merged the vertices into is also exiting.
        if edge
            .vertices
            .into_iter()
            .all(|vertex| !frontier.contains(&vertex))
        {
            new_node.handle_component_exit(edge.vertices[0], remaining_edges, frontier)?;
        }

        Ok(new_node)
    }
}

/// A node pointed to by an edge coming out of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum NodeRef {
    /// The 0-node.
    Zero,
    /// The 1-node.
    One,
    /// The node at index `index` in the next row of the ZDD.
    NextRow { index: usize },
}

/// One of the constant nodes (the 0-node or the 1-node). Used as an `Err`
/// variant for when an edge should point to one of these instead of new node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ConstantNode {
    /// The 0-node.
    Zero,
    /// The 1-node.
    One,
}

/// A vertex of a cuboid.
///
/// Unlike in `Net` and `Surface`, this isn't the middle of a square on the
/// surface of a cuboid; it's the corner.
/// To be clear, it's not just the corners of the entire cuboid, it still
/// includes the corners of all the grid squares on the cuboid's surface.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Vertex {
    /// The cuboid this is a vertex of.
    cuboid: Cuboid,
    /// Which face of the cuboid this vertex lies on.
    ///
    /// Some vertices lie on the edge of 2 (or 3) faces. In that case, we just
    /// pick one of them to specify here.
    face: Face,
    /// The position of this vertex on the face.
    pos: Pos,
}

/// A edge between two vertices of a cuboid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
struct Edge {
    /// The indices of the two endpoints of this edge.
    ///
    /// Sorted from smallest to largest.
    vertices: [usize; 2],
}

impl Edge {
    fn new(v1: usize, v2: usize) -> Self {
        let mut vertices = [v1, v2];
        vertices.sort_unstable();
        Self { vertices }
    }
}

impl Vertex {
    /// Returns the position of this vertex on the given face, if it's on that
    /// face.
    fn pos_on_face(&self, face: Face) -> Option<Pos> {
        let pos_in_direction = |direction| {
            let (new_face, turns) = self.face.adjacent_in(direction);
            let mut new_pos = self.pos;
            // First, rotate our position so that it's the right way around for the new
            // face.
            let mut size = self.cuboid.face_size(self.face);
            for _ in 0..turns.rem_euclid(4) {
                size.rotate(1);
                new_pos = Pos {
                    x: new_pos.y,
                    y: size.height - new_pos.x,
                };
            }
            // Then adjust the part of the position where we entered.
            let entry_direction = direction.turned(turns);
            let size = self.cuboid.face_size(new_face);
            match entry_direction {
                Left => new_pos.x = size.width,
                Up => new_pos.y = 0,
                Right => new_pos.x = 0,
                Down => new_pos.y = size.height,
            }
            new_pos
        };

        if self.face == face {
            Some(self.pos)
        } else if self.pos.x == 0 && self.face.adjacent_in(Left).0 == face {
            Some(pos_in_direction(Left))
        } else if self.pos.y == self.cuboid.face_size(self.face).height
            && self.face.adjacent_in(Up).0 == face
        {
            Some(pos_in_direction(Up))
        } else if self.pos.x == self.cuboid.face_size(self.face).width
            && self.face.adjacent_in(Right).0 == face
        {
            Some(pos_in_direction(Right))
        } else if self.pos.y == 0 && self.face.adjacent_in(Down).0 == face {
            Some(pos_in_direction(Down))
        } else {
            None
        }
    }

    /// Returns the list of vertices adjacent to this one.
    fn neighbours(&self) -> Vec<Vertex> {
        let mut neighbours = Vec::new();
        // Go through the version of this vertex on every face and find its neighbours
        // on that face.
        for (face, pos) in [Bottom, West, North, East, South, Top]
            .into_iter()
            .filter_map(|face| Some((face, self.pos_on_face(face)?)))
        {
            if pos.x > 0 {
                neighbours.push(Vertex {
                    cuboid: self.cuboid,
                    face,
                    pos: Pos {
                        x: pos.x - 1,
                        y: pos.y,
                    },
                })
            }
            if pos.x < self.cuboid.face_size(face).width {
                neighbours.push(Vertex {
                    cuboid: self.cuboid,
                    face,
                    pos: Pos {
                        x: pos.x + 1,
                        y: pos.y,
                    },
                })
            }

            if pos.y > 0 {
                neighbours.push(Vertex {
                    cuboid: self.cuboid,
                    face,
                    pos: Pos {
                        x: pos.x,
                        y: pos.y - 1,
                    },
                })
            }
            if pos.y < self.cuboid.face_size(face).height {
                neighbours.push(Vertex {
                    cuboid: self.cuboid,
                    face,
                    pos: Pos {
                        x: pos.x,
                        y: pos.y + 1,
                    },
                })
            }
        }
        neighbours
    }
}

#[test]
fn pos_on_face() {
    let cuboid = Cuboid::new(1, 1, 1);
    let vertex = Vertex {
        cuboid,
        face: Bottom,
        pos: Pos::new(0, 0),
    };
    assert_eq!(vertex.pos_on_face(Bottom), Some(Pos::new(0, 0)));
    assert_eq!(vertex.pos_on_face(West), Some(Pos::new(0, 0)));
    assert_eq!(vertex.pos_on_face(North), None);
    assert_eq!(vertex.pos_on_face(East), None);
    assert_eq!(vertex.pos_on_face(South), Some(Pos::new(1, 0)));
    assert_eq!(vertex.pos_on_face(Top), None);

    let vertex = Vertex {
        cuboid,
        face: Top,
        pos: Pos::new(0, 0),
    };
    assert_eq!(vertex.pos_on_face(Bottom), None);
    assert_eq!(vertex.pos_on_face(West), Some(Pos::new(1, 1)));
    assert_eq!(vertex.pos_on_face(North), Some(Pos::new(0, 1)));
    assert_eq!(vertex.pos_on_face(East), None);
    assert_eq!(vertex.pos_on_face(South), None);
    assert_eq!(vertex.pos_on_face(Top), Some(Pos::new(0, 0)));
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        other.pos_on_face(self.face) == Some(self.pos)
    }
}

impl Eq for Vertex {}

/// Entrypoint for a thread that computes one of the rows of a ZDD.
fn row_thread(
    edges: &[Edge],
    frontier: &[usize],
    progress_bar: ProgressBar,
    index: usize,
    receiver: mpsc::Receiver<NodeState>,
    sender: Option<mpsc::Sender<NodeState>>,
) -> Vec<Node> {
    let edge = edges[index];

    // The row of nodes that this thread is building up.
    let mut row: Vec<Node> = Vec::new();

    // Store the the `NodeState`s we're passing off to the next row for
    // de-duplication.
    //
    // TODO: this might be able to be solved by storing one of the paths through the
    // ZDD that you can take to get to a `NodeState`, probably as a `u128` bitfield
    // of taken edges, and then recreating it on the fly. You don't actually
    // need access to the ZDD to do that, just to the list of edges. If computing
    // that's too slow, you could maybe use an LRU cache or something.
    // Since we're storing hashes anyway, `NoopHasher` is a hasher which just
    // returns an input `u64` verbatim.
    let mut yielded_states: FxHashMap<NodeState, usize> = FxHashMap::default();
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
                    NodeRef::NextRow { index }
                } else {
                    // We've found a new `NodeState`, send it to the next row.
                    sender
                        .as_ref()
                        .expect("node in last row doesn't lead to a constant node")
                        .send(new_node.clone())
                        .unwrap();
                    let index = yielded_states.len();
                    yielded_states.insert(new_node, index);
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
        // First create the 0-edge, which represents not adding this edge to the graph.
        // We first create a new `NodeState` representing this, and then check
        // `state_hashes` to see if it already exists. If it does, use that one instead.
        let zero_edge = handle_state(state.zero_edge(edge, &edges[index + 1..], frontier));

        // Then we do the same thing for the 1-edge, which represents adding this edge
        // to the graph.
        let one_edge = handle_state(state.one_edge(edge, &edges[index + 1..], frontier));

        // Now add the new node to the row.
        row.push(Node::new(zero_edge, one_edge));

        // Updating the progress bar is surprisingly slow thanks to some mutex stuff, so
        // only do it every 50ms.
        if progress_last_updated.elapsed() > Duration::from_millis(50) {
            progress_bar.set_position(row.len().try_into().unwrap());
            // For each vertex, `NodeState` stores 1 byte + 1 bit: the component it's in and whether it's pendant.
            // I'm representing
            let node_state_size = mem::size_of::<NodeState>()
                + state.components.len()
                + (state.components.len() + usize::BITS as usize - 1) / usize::BITS as usize
                    * mem::size_of::<usize>();
            progress_bar.set_message(format!(
                "mem. usage: nodes: {}, states: {}",
                HumanBytes((row.len() * mem::size_of::<Node>()) as u64),
                HumanBytes((num_yielded_states.get() * node_state_size) as u64)
            ));
            progress_last_updated = Instant::now();
        }
    }

    progress_bar.set_length(row.len().try_into().unwrap());
    progress_bar.finish_with_message("done");

    row
}

impl Zdd {
    pub fn construct(cuboid: Cuboid) -> Self {
        // Enumerate all the vertices and edges of the cuboid.
        let (vertices, edges) = Self::compute_geometry(cuboid);
        let num_vertices: u8 = vertices.len().try_into().expect("too many vertices");

        // Create the list of 'frontiers', which is basically the list of vertices which
        // are still 'relevant' at a given row of the ZDD.
        // Each row of the ZDD represents an edge which is or isn't added to the graph.
        // I'm gonna say that a vertex is 'considered' in a row if it's one of the
        // endpoints of the edge potentially being added that row.
        // The frontier for a row is the list of vertices which have been considered in
        // previous rows or are being considered for the first time this row, and which
        // will be considered again in future rows.
        let mut frontiers = vec![vec![]; edges.len()];
        for v in 0..vertices.len() {
            // Find the first and last time this vertex is considered, and then add the
            // vertex to the frontiers between them.
            let first_occurence = edges.iter().position(|e| e.vertices.contains(&v)).unwrap();
            let last_occurence = edges.iter().rposition(|e| e.vertices.contains(&v)).unwrap();
            // Note: we specifically don't include the last occurence, since that's
            // precisely when we need to know that a vertex is done.
            for frontier in &mut frontiers[first_occurence..last_occurence] {
                frontier.push(v);
            }
        }

        // Create a progress bar for each thread, which is initially hidden until the
        // thread gets its first `NodeState` to process.
        let progress = MultiProgress::new();
        let style =
            ProgressStyle::with_template("{prefix} - {wide_msg} {human_pos} nodes {spinner}")
                .unwrap();
        let progress_bars: Vec<_> = (0..edges.len())
            .map(|i| {
                let bar = ProgressBar::new_spinner()
                    .with_prefix(format!("row {}", i + 1))
                    .with_style(style.clone());
                progress.add(bar.clone());
                bar
            })
            .collect();

        let rows = thread::scope(|s| {
            let (senders, receivers): (Vec<_>, Vec<_>) = (0..edges.len())
                .map(|_| mpsc::channel::<NodeState>())
                .unzip();
            // Extract the first sender, which is used to send the initial `NodeState` to
            // the thread handling the first row.
            let mut senders = senders.into_iter();
            let first_sender = senders.next().unwrap();
            let handles: Vec<_> = zip(senders.map(Some).chain([None]), receivers)
                .enumerate()
                .map(|(index, (sender, receiver))| {
                    let edges = &edges;
                    let frontiers = &frontiers[index];
                    let progress_bar = progress_bars[index].clone();
                    thread::Builder::new()
                        .name(format!("row {}", index + 1))
                        .spawn_scoped(s, move || {
                            row_thread(edges, frontiers, progress_bar, index, receiver, sender)
                        })
                        .unwrap()
                })
                .collect();

            first_sender.send(NodeState::new(num_vertices)).unwrap();
            drop(first_sender);

            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });

        Self {
            cuboid,
            edges,
            vertices,
            rows,
        }
    }

    /// Computes the vertices and edges of a cuboid.
    fn compute_geometry(cuboid: Cuboid) -> (Vec<Vertex>, Vec<Edge>) {
        // Enumerate all the vertices of the cuboid, starting with the corners..
        let mut vertices: Vec<_> = [
            (Bottom, Pos::new(0, 0)),
            (Bottom, Pos::new(0, cuboid.depth)),
            (Bottom, Pos::new(cuboid.width, 0)),
            (Bottom, Pos::new(cuboid.width, cuboid.depth)),
            (Top, Pos::new(0, 0)),
            (Top, Pos::new(0, cuboid.depth)),
            (Top, Pos::new(cuboid.width, 0)),
            (Top, Pos::new(cuboid.width, cuboid.depth)),
        ]
        .into_iter()
        .map(|(face, pos)| Vertex { cuboid, face, pos })
        .collect();

        for face in [Bottom, West, North, East, South, Top] {
            let size = cuboid.face_size(face);
            for x in 0..=size.width {
                for y in 0..=size.height {
                    let vertex = Vertex {
                        cuboid,
                        face,
                        pos: Pos { x, y },
                    };
                    if !vertices.contains(&vertex) {
                        vertices.push(vertex);
                    }
                }
            }
        }

        // Enumerate all the edges of the cuboid.
        let mut edges = Vec::new();
        for (i, v1) in vertices.iter().enumerate() {
            for v2 in v1.neighbours() {
                let j = vertices.iter().position(|&vertex| vertex == v2).unwrap();
                let edge = Edge::new(i, j);
                if !edges.contains(&edge) {
                    edges.push(edge);
                }
            }
        }

        (vertices, edges)
    }

    pub fn cuboid_info(cuboid: Cuboid) -> (usize, usize) {
        let (vertices, edges) = Self::compute_geometry(cuboid);

        (vertices.len(), edges.len())
    }

    pub fn vertex_indices(&self) -> FxHashMap<(Face, Pos), usize> {
        let mut vertex_indices = FxHashMap::default();
        for (i, vertex) in self.vertices.iter().copied().enumerate() {
            for face in [Bottom, West, North, East, South, Top] {
                if let Some(pos) = vertex.pos_on_face(face) {
                    vertex_indices.insert((face, pos), i);
                }
            }
        }
        vertex_indices
    }

    /// Returns the number of developments this ZDD represents.
    pub fn size(&self) -> usize {
        // Set up a cache for the number of developments represented by each node.
        let mut cache: Vec<_> = self.rows.iter().map(|row| vec![None; row.len()]).collect();

        // Then go through and recursively calculate the sizes of all the nodes.
        fn size(zdd: &Zdd, cache: &mut [Vec<Option<usize>>], row: usize, edge: NodeRef) -> usize {
            match edge {
                NodeRef::Zero => 1,
                NodeRef::One => 0,
                NodeRef::NextRow { index } => {
                    if cache[row + 1][index].is_none() {
                        let node = &zdd.rows[row + 1][index];
                        cache[row + 1][index] = Some(
                            size(zdd, cache, row + 1, node.zero_edge())
                                + size(zdd, cache, row + 1, node.one_edge()),
                        )
                    }
                    cache[row + 1][index].unwrap()
                }
            }
        }

        let node = &self.rows[0][0];
        let result = size(self, &mut cache, 0, node.zero_edge())
            + size(self, &mut cache, 0, node.one_edge());
        // for row in &cache[1..] {
        //     for size in row {
        //         print!("{} ", size.unwrap());
        //     }
        //     println!();
        // }
        result
    }

    /// Returns an iterator over the set of sets of cut edges this ZDD
    /// represents.
    fn edges(&self) -> EdgeIter<'_> {
        EdgeIter::new(self)
    }

    /// Returns an iterator over the list of nets that this ZDD represents.
    pub fn nets(&self) -> impl Iterator<Item = Net> + '_ {
        // Create a mapping from face and position to the index of the vertex with that
        // position, to avoid having to use `pos_on_face` and iterating through the list
        // of vertices every time.
        let vertex_indices = self.vertex_indices();

        Iterator::map(self.edges(), move |edges| {
            net_from_edges(self.cuboid, edges, &vertex_indices)
        })
    }

    /// Returns a parallel iterator over the list of nets that this ZDD
    /// represents.
    pub fn par_nets(&self) -> impl ParallelIterator<Item = Net> + '_ {
        // Create a mapping from face and position to the index of the vertex with that
        // position, to avoid having to use `pos_on_face` and iterating through the list
        // of vertices every time.
        let vertex_indices = self.vertex_indices();

        ParallelIterator::map(self.edges(), move |edges| {
            net_from_edges(self.cuboid, edges, &vertex_indices)
        })
    }

    /// Returns the (approximate) amount of space this ZDD takes up on the heap.
    ///
    /// Approximate because I couldn't be bothered to count anything other than
    /// the nodes, which take up basically all the space anyway.
    pub fn heap_size(&self) -> usize {
        let nodes: usize = self.rows.iter().map(|row| row.len()).sum();
        nodes * mem::size_of::<Node>()
    }
}

#[derive(Clone)]
struct EdgeIter<'a> {
    zdd: &'a Zdd,
    /// The current path through the ZDD we're looking at.
    /// Between calls to `next`, this is the path we're going to try
    /// next.
    stack: Vec<(Node, bool)>,
    /// If this is being used in parallel, the index in `stack` at which this
    /// iterator was split off. In other words, it's the point in `stack` we
    /// shouldn't backtrack past, because that's other threads' jobs.
    /// This is implemented semi-crudely; `advance` ignores this and keeps
    /// backtracking anyway, but then `endpoint` returns `None` once the split
    /// point has been popped off, so it works out.
    split_point: usize,
}

impl<'a> EdgeIter<'a> {
    fn new(zdd: &'a Zdd) -> Self {
        Self {
            stack: vec![(zdd.rows[0][0], false)],
            zdd,
            split_point: 0,
        }
    }

    /// Returns the node that the path currently leads to.
    fn endpoint(&self) -> Option<NodeRef> {
        if self.stack.len() <= self.split_point {
            return None;
        }
        let (node, edge) = self.stack.last()?;
        Some(match edge {
            false => node.zero_edge(),
            true => node.one_edge(),
        })
    }

    /// Move `stack` forward to the next valid path, without actually
    /// looking at what it leads to.
    fn advance(&mut self) {
        if let Some(NodeRef::NextRow { index }) = self.endpoint() {
            // Add the new node to the stack.
            let node = self.zdd.rows[self.stack.len()][index];
            self.stack.push((node, false));
        } else {
            // Backtrack until we find a node that we haven't already tried both edges of.
            while matches!(self.stack.last(), Some((_, true))) {
                self.stack.pop();
            }
            // Then try its other edge, assuming we haven't exhausted all the possibilities
            // making it `None`.
            if let Some((_, edge)) = self.stack.last_mut() {
                *edge = true;
            }
        }
    }
}

impl Iterator for EdgeIter<'_> {
    type Item = Vec<Edge>;

    fn next(&mut self) -> Option<Self::Item> {
        // Keep traversing the ZDD until we get to a 1-node.
        // Also, if `self.endpoint()` returns `None`, we've exhausted all possibilities,
        // so propagate the `None`.
        while self.endpoint()? != NodeRef::One {
            self.advance();
        }

        // If we broke out of the loop, we've found a path to a 1-node.
        let result = self
            .stack
            .iter()
            .enumerate()
            .filter_map(|(i, (_, included))| included.then_some(self.zdd.edges[i]))
            .collect();

        // Advance the path for next time.
        self.advance();

        Some(result)
    }
}

impl UnindexedProducer for EdgeIter<'_> {
    type Item = Vec<Edge>;

    fn split(mut self) -> (Self, Option<Self>) {
        if self.endpoint().is_none() {
            // This iterator's already exhausted, don't bother splitting it.
            return (self, None);
        }

        let old_split_point = self.split_point;

        // Move upwards through the stack until we find a node we haven't tried both
        // branches of yet.
        while matches!(self.stack.get(self.split_point), Some((_, true))) {
            self.split_point += 1;
        }
        if self.stack.get(self.split_point).is_none() {
            // We've gone past the end of how far we've extended the stack through the ZDD,
            // so we have to extend it by one more node so that we can split on its two
            // edges.
            // First we have to subtract 1 from the split point so that `endpoint` works,
            // though.
            self.split_point -= 1;
            match self.endpoint() {
                Some(NodeRef::NextRow { index }) => {
                    // Add the new node to the stack.
                    let node = self.zdd.rows[self.stack.len()][index];
                    self.stack.push((node, false));
                    self.split_point += 1;
                }
                // If everything up to here was a 1-edge, and the next node is a constant node, that
                // means that this iterator's already exhausted, and just hasn't backtracked yet.
                // Don't bother splitting.
                _ => {
                    self.split_point = old_split_point;
                    return (self, None);
                }
            }
        }

        // Finally, we can use the two edges of the previous split point as the split
        // points for the two new iterators.
        let (split_point, _) = self.stack[self.split_point];
        let (NodeRef::NextRow { index: index1 }, NodeRef::NextRow { index: index2 })
            = (split_point.zero_edge(), split_point.one_edge()) else {
            // We could try to traverse further to find a working split point, but what do
            // we do if we find a 1-node that we should be yielding?
            // It's a lot of hassle for what's probably a very rare case.
            self.split_point = old_split_point;
            return (self, None);
        };
        let mut other = self.clone();

        if self.split_point == self.stack.len() - 1 {
            // Add the split point to the existing iterator if it's not already there.
            // (note: we know that this is already the 0-edge in the case where it's already
            // there, since we made sure of that in the big loop above.)
            let node = self.zdd.rows[self.stack.len()][index1];
            self.stack.push((node, false));
        }
        self.split_point += 1;

        // Get rid of the other iterator's stack beyond the old split point and then add
        // its new split point.
        other
            .stack
            .resize_with(other.split_point + 1, || unreachable!());
        // Update the previous split point to say that the 1-edge was taken.
        other.stack.last_mut().unwrap().1 = true;

        let node = other.zdd.rows[other.stack.len()][index2];
        other.stack.push((node, false));
        other.split_point += 1;

        (self, Some(other))
    }

    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        folder.consume_iter(self)
    }
}

impl ParallelIterator for EdgeIter<'_> {
    type Item = Vec<Edge>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge_unindexed(self, consumer)
    }
}

fn net_from_edges(
    cuboid: Cuboid,
    mut edges: Vec<Edge>,
    vertex_indices: &FxHashMap<(Face, Pos), usize>,
) -> Net {
    edges.sort_unstable();
    let mut net = Net::for_cuboids(&[cuboid]);
    let middle_x = net.width() / 2;
    let middle_y = net.height() / 2;

    // Start in the middle of the net, and traverse out as far as we can until we
    // run into edges.
    let pos = Pos {
        x: middle_x,
        y: middle_y,
    };
    let face_pos = FacePos::new(cuboid);

    net.set(pos, true);
    let mut area = 1;

    // Create a queue of places to try extruding squares.
    let mut queue: VecDeque<_> = [Left, Up, Right, Down]
        .into_iter()
        .map(|direction| (pos, face_pos, direction))
        .collect();

    while let Some((pos, face_pos, direction)) = queue.pop_front() {
        let new_pos = pos
            .moved_in(direction, net.size())
            .expect("this should have already been filtered out before adding to queue");
        if net.filled(new_pos) {
            continue;
        }
        let new_face_pos = face_pos.moved_in(direction);

        // Construct the edge that would stop this from happening.

        // First, figure out the endpoints on this face of the edge.

        // Also, a note on indexing here: the vertex with the same `Pos` value as a face
        // position actually represents the bottom left corner of the square at that
        // face position. So, you can add 1 to the x and y of the position to get to the
        // other corners of the square.
        let (pos1, pos2) = match direction.turned(face_pos.orientation) {
            Left => (
                Pos {
                    x: face_pos.pos.x,
                    y: face_pos.pos.y,
                },
                Pos {
                    x: face_pos.pos.x,
                    y: face_pos.pos.y + 1,
                },
            ),
            Up => (
                Pos {
                    x: face_pos.pos.x,
                    y: face_pos.pos.y + 1,
                },
                Pos {
                    x: face_pos.pos.x + 1,
                    y: face_pos.pos.y + 1,
                },
            ),
            Right => (
                Pos {
                    x: face_pos.pos.x + 1,
                    y: face_pos.pos.y,
                },
                Pos {
                    x: face_pos.pos.x + 1,
                    y: face_pos.pos.y + 1,
                },
            ),
            Down => (
                Pos {
                    x: face_pos.pos.x,
                    y: face_pos.pos.y,
                },
                Pos {
                    x: face_pos.pos.x + 1,
                    y: face_pos.pos.y,
                },
            ),
        };

        // Then find the vertices at those positions.
        let v1 = vertex_indices[&(face_pos.face, pos1)];
        let v2 = vertex_indices[&(face_pos.face, pos2)];
        let edge = Edge::new(v1, v2);

        // Finally, check if that edge that would stop us exists. If not, we add the
        // square.
        if edges.binary_search(&edge).is_ok() {
            continue;
        }

        net.set(new_pos, true);
        area += 1;
        if area > cuboid.surface_area() {
            panic!("Invalid edge set found: {edges:?}");
        }

        // Then add the new spots we could extrude squares to the queue.
        for direction in [Left, Up, Right, Down] {
            if let Some(newer_pos) = new_pos.moved_in(direction, net.size()) {
                if !net.filled(newer_pos) {
                    queue.push_back((new_pos, new_face_pos, direction));
                }
            }
        }
    }

    net.shrink()
}
