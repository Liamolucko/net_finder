use std::{
    cell::Cell,
    collections::{HashSet, VecDeque},
    hash::{Hash, Hasher},
    iter::zip,
    mem,
    sync::mpsc,
    thread::{self, ScopedJoinHandle},
    time::{Duration, Instant},
};

use indicatif::{HumanBytes, MultiProgress, ProgressBar, ProgressStyle};
use once_cell::unsync::OnceCell;
use rustc_hash::FxHashMap;

use crate::Cuboid;

use super::{ConstantNode, Edge, Node, NodeRef, Zdd, MAX_EDGES, MAX_VERTICES};

thread_local! {
    // I hate to use global state, but this doesn't have any risk of colliding since it's local to each row thread, which we completely control.
    // We need it to implement `EdgeSet`.
    /// A tuple of the number of vertices and the edges in the cuboid we're constructing a ZDD for.
    static GEOMETRY: OnceCell<(u8, Vec<Edge>)> = OnceCell::new();
}

/// The state of the graph represented by a node in a ZDD.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NodeState {
    /// Data about each vertex in the graph.
    /// The MSB represents whether the vertex is pendant, and the rest of the
    /// byte is the index of the component the vertex is in.
    data: heapless::Vec<u8, { MAX_VERTICES as usize }>,
}

impl NodeState {
    /// Create a new node without any info filled in.
    fn new(num_vertices: u8) -> Self {
        Self {
            data: (0..num_vertices).collect(),
        }
    }

    fn component(&self, vertex: usize) -> u8 {
        self.data[vertex] & 0b01111111
    }

    fn set_component(&mut self, vertex: usize, component: u8) {
        debug_assert!(component < MAX_VERTICES);
        // Clear the old value, then OR in the new value.
        self.data[vertex] &= 0b10000000;
        self.data[vertex] |= component;
    }

    fn pendant(&self, vertex: usize) -> bool {
        self.data[vertex] & 0b10000000 != 0
    }

    fn set_pendant(&mut self, vertex: usize, pendant: bool) {
        self.data[vertex] &= 0b01111111;
        self.data[vertex] |= (pendant as u8) << 7;
    }

    fn size(&self, vertex: usize) -> usize {
        let component = self.component(vertex);
        (0..self.data.len())
            .filter(|&other_vertex| self.component(other_vertex) == component)
            .count()
    }

    /// Adds an edge to this `NodeState` without performing any kind of validation.
    fn add_edge(&mut self, edge: Edge) {
        // First we update whether or not the vertices are pendant.
        for vertex in edge.vertices {
            if self.size(vertex) == 1 {
                // This vertex is getting its first edge added to it, and becoming pendant.
                self.set_pendant(vertex, true);
            } else if self.pendant(vertex) {
                // This pendant vertex is getting a second edge added to it, and is no longer
                // pendant.
                self.set_pendant(vertex, false);
            }
        }

        // Then we merge the two components that the endpoints of the edge are in.
        let comp1 = self.component(edge.vertices[0]);
        let comp2 = self.component(edge.vertices[1]);
        let min_comp = u8::min(comp1, comp2);
        let max_comp = u8::max(comp1, comp2);

        // The minimum of the two component indices is the new index for the merged
        // component.
        let new_index = min_comp;

        for vertex in 0..self.data.len() {
            // Set the component index for any vertices with the old component index to the
            // new index.
            if self.component(vertex) == max_comp {
                self.set_component(vertex, new_index);
            }
        }
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
            if self.pendant(vertex) {
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
        frontier: impl IntoIterator<Item = usize> + Clone,
    ) -> Result<(), ConstantNode> {
        if !frontier
            .clone()
            .into_iter()
            .any(|other_vertex| self.component(other_vertex) == self.component(vertex))
        {
            // If there aren't any vertices left in the frontier that are in the same
            // connected component as this vertex, this component is now also never going to
            // be modified again.

            // If the component is of size > 1, it's _the_ connected component of the graph,
            // since there's only allowed to be one with size > 1.
            if self.size(vertex) > 1 {
                if frontier.into_iter().any(|vertex| self.size(vertex) > 1) {
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

        // Then check if the exits of these vertices result in the exit of their
        // connected components.
        match (
            frontier.contains(&edge.vertices[0]),
            frontier.contains(&edge.vertices[1]),
        ) {
            (false, false) => {
                self.handle_component_exit(
                    edge.vertices[0],
                    remaining_edges,
                    frontier.iter().copied().chain([edge.vertices[1]]),
                )?;
                self.handle_component_exit(
                    edge.vertices[1],
                    remaining_edges,
                    frontier.iter().copied(),
                )?;
            }
            (false, true) => {
                self.handle_component_exit(
                    edge.vertices[0],
                    remaining_edges,
                    frontier.iter().copied(),
                )?;
            }
            (true, false) => {
                self.handle_component_exit(
                    edge.vertices[1],
                    remaining_edges,
                    frontier.iter().copied(),
                )?;
            }
            (true, true) => {}
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
        if self.component(edge.vertices[0]) == self.component(edge.vertices[1]) {
            // Adding this edge would create a cycle, which isn't allowed, so
            // discard this possibility by pointing to the zero-edge.
            return Err(ConstantNode::Zero);
        }

        let mut new_node = self.clone();

        // Actually update the info in `new_node` to account for this edge being added.
        new_node.add_edge(edge);

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
            new_node.handle_component_exit(
                edge.vertices[0],
                remaining_edges,
                frontier.iter().copied(),
            )?;
        }

        Ok(new_node)
    }
}

/// A sort of compressed encoding for `NodeState` which just stores one of the paths through the ZDD that can be taken to get to it.
///
/// That can also just be interpreted as a set of edges.
#[derive(Debug, Clone, Copy)]
pub(super) struct NodePath {
    bits: [u8; (MAX_EDGES + 7) / 8],
}

impl NodePath {
    /// Returns a new `NodePath` with no edges set.
    pub(super) fn new() -> Self {
        Self {
            bits: [0; (MAX_EDGES + 7) / 8],
        }
    }

    /// Evaluates this `NodePath` into a `NodeState`
    fn evaluate(self) -> NodeState {
        GEOMETRY.with(|cell| {
            let &(num_vertices, ref edges) = cell.get().unwrap();
            let mut res = NodeState::new(num_vertices);
            edges
                .iter()
                .copied()
                .enumerate()
                .filter(|(i, _)| self.bits[i / 8] & (1 << (i % 8)) != 0)
                .for_each(|(_, edge)| res.add_edge(edge));
            res
        })
    }

    /// Returns a copy of `self` with the edge at the given index set.
    fn with_edge(mut self, edge: usize) -> Self {
        self.bits[edge / 8] |= 1 << (edge % 8);
        self
    }
}

impl PartialEq for NodePath {
    fn eq(&self, other: &Self) -> bool {
        self.evaluate() == other.evaluate()
    }
}

impl Eq for NodePath {}

impl Hash for NodePath {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.evaluate().hash(state)
    }
}

/// Entrypoint for a thread that computes one of the rows of a ZDD.
pub(super) fn row_thread(
    num_vertices: u8,
    edges: &[Edge],
    frontier: &[usize],
    progress_bars: &[ProgressBar],
    index: usize,
    receiver: mpsc::Receiver<NodePath>,
    sender: Option<mpsc::Sender<NodePath>>,
) -> Vec<Node> {
    GEOMETRY.with(|cell| {
        cell.set((num_vertices, edges.to_vec())).unwrap();
    });

    let edge = edges[index];
    let progress_bar = progress_bars[index].clone();
    let next_progress_bar = progress_bars.get(index + 1).cloned();

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
    let mut yielded_states: FxHashMap<NodePath, usize> = FxHashMap::default();
    // A hack to work around not being able to access `yielded_states.len()` since
    // it's exclusively borrowed by `handle_state`.
    let num_yielded_states = Cell::new(0);

    // Create a helper function that takes the result of `NodeState::zero_edge` or
    // `NodeState::one_edge` and turns it into a `NodeRef`, handling all the
    // de-duping and such.
    let mut handle_state = |result: Result<NodeState, ConstantNode>, path: NodePath| -> NodeRef {
        match result {
            Ok(_) => {
                if let Some(&index) = yielded_states.get(&path) {
                    NodeRef::NextRow { index }
                } else {
                    // We've found a new `NodeState`, send it to the next row.
                    sender
                        .as_ref()
                        .expect("node in last row doesn't lead to a constant node")
                        .send(path)
                        .unwrap();
                    let index = yielded_states.len();
                    yielded_states.insert(path, index);
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
    for path in receiver {
        let state = path.evaluate();

        // First create the 0-edge, which represents not adding this edge to the graph.
        // We first create a new `NodeState` representing this, and then check
        // `state_hashes` to see if it already exists. If it does, use that one instead.
        let zero_edge = handle_state(state.zero_edge(edge, &edges[index + 1..], frontier), path);

        // Then we do the same thing for the 1-edge, which represents adding this edge
        // to the graph.
        let path = path.with_edge(index);
        let one_edge = handle_state(state.one_edge(edge, &edges[index + 1..], frontier), path);

        // Now add the new node to the row.
        row.push(Node::new(zero_edge, one_edge));

        // Updating the progress bar is surprisingly slow thanks to some mutex stuff, so
        // only do it every 50ms.
        if progress_last_updated.elapsed() > Duration::from_millis(50) {
            progress_bar.set_position(row.len().try_into().unwrap());
            progress_bar.set_message(format!(
                "mem. usage: nodes: {}, states: {}, queue: {}",
                HumanBytes((row.len() * mem::size_of::<Node>()) as u64),
                HumanBytes((num_yielded_states.get() * mem::size_of::<NodePath>()) as u64),
                HumanBytes(
                    progress_bar
                        .length()
                        .unwrap()
                        .saturating_sub(row.len() as u64)
                        * mem::size_of::<NodePath>() as u64
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

impl Zdd {
    pub fn construct(cuboid: Cuboid) -> Self {
        // Enumerate all the vertices and edges of the cuboid.
        let (vertices, edges) = Self::compute_geometry(cuboid);
        assert!(vertices.len() <= MAX_VERTICES.into());
        assert!(edges.len() <= MAX_EDGES);
        let num_vertices: u8 = vertices.len().try_into().unwrap();

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
        let style = ProgressStyle::with_template(
            "{prefix} - {msg} {wide_bar} {human_pos} / {human_len} nodes",
        )
        .unwrap();
        let progress_bars: Vec<_> = (0..edges.len())
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
            let (senders, receivers): (Vec<_>, Vec<_>) = (0..edges.len())
                .map(|_| mpsc::channel::<NodePath>())
                .unzip();
            // Extract the first sender, which is used to send the initial `NodeState` to
            // the thread handling the first row.
            let mut senders = senders.into_iter();
            let first_sender = senders.next().unwrap();
            first_sender.send(NodePath::new()).unwrap();
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
                let edges = &edges;
                let frontiers = &frontiers[index];
                let progress_bars = &progress_bars;
                let handle = thread::Builder::new()
                    .name(format!("row {}", index + 1))
                    .spawn_scoped(s, move || {
                        row_thread(
                            num_vertices,
                            edges,
                            frontiers,
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

        Self {
            cuboid,
            edges,
            vertices,
            rows,
        }
    }
}
