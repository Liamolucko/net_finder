use std::cell::Cell;
use std::collections::{HashSet, VecDeque};
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::iter::zip;
use std::mem;
use std::num::NonZeroU8;
use std::sync::mpsc;
use std::thread::{self, ScopedJoinHandle};
use std::time::{Duration, Instant};

use indicatif::{HumanBytes, MultiProgress, ProgressBar, ProgressStyle};
use rustc_hash::FxHashMap;

use crate::Cuboid;

use super::geometry::{rotations_for_cuboid, Edge};
use super::{ConstantNode, Node, NodeRef, Zdd, MAX_EDGES, MAX_VERTICES};

/// The state of the graph represented by a node in a ZDD.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NodeState {
    /// Data about each vertex in the graph.
    ///
    /// The MSB represents whether the vertex is pendant, and the rest of the
    /// byte is the index of the component the vertex is in, or 0 if it isn't in
    /// one.
    ///
    /// Component indices are a bit special - they need to be able to be
    /// determined deterministically for a given set of vertices that make up a
    /// component. So, the index of a component is always the the index of the
    /// vertex in the component with the lowest index, plus one since zero
    /// means not being in a component.
    data: [u8; MAX_VERTICES as usize],
    /// The number of vertices in the graph, which effectively also functions as
    /// `data.len()`.
    num_vertices: u8,
}

impl Debug for NodeState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for vertex in 0..self.num_vertices {
            list.entry(&(self.pendant(vertex), self.component(vertex)));
        }
        list.finish()
    }
}

impl NodeState {
    /// Create a new node without any info filled in.
    fn new(num_vertices: u8) -> Self {
        assert!(num_vertices <= MAX_VERTICES);
        Self {
            data: [0; MAX_VERTICES as usize],
            num_vertices,
        }
    }

    /// Creates a component index derived from a vertex index, which is just the
    /// index + 1.
    fn component_index(vertex: u8) -> NonZeroU8 {
        NonZeroU8::new(vertex + 1).unwrap()
    }

    /// Returns the connected component this vertex is in, or `None` if it's not
    /// connected to any other vertices.
    fn component(&self, vertex: u8) -> Option<NonZeroU8> {
        NonZeroU8::new(self.data[usize::from(vertex)] & 0b01111111)
    }

    fn set_component(&mut self, vertex: u8, component: NonZeroU8) {
        debug_assert!(component.get() < MAX_VERTICES);
        // Clear the old value, then OR in the new value.
        self.data[usize::from(vertex)] &= 0b10000000;
        self.data[usize::from(vertex)] |= component.get();
    }

    fn pendant(&self, vertex: u8) -> bool {
        self.data[usize::from(vertex)] & 0b10000000 != 0
    }

    fn set_pendant(&mut self, vertex: u8, pendant: bool) {
        self.data[usize::from(vertex)] &= 0b01111111;
        self.data[usize::from(vertex)] |= (pendant as u8) << 7;
    }

    fn change_component_index(&mut self, old_index: NonZeroU8, new_index: NonZeroU8) {
        for vertex in 0..self.num_vertices {
            if self.component(vertex) == Some(old_index) {
                self.set_component(vertex, new_index);
            }
        }
    }

    /// Adds an edge to this `NodeState` without performing any kind of
    /// validation.
    fn add_edge(&mut self, edge: Edge) {
        // First we update whether or not the vertices are pendant.
        for vertex in edge.vertices {
            if self.component(vertex).is_none() {
                // This vertex is getting its first edge added to it, and becoming pendant.
                self.set_pendant(vertex, true);
            } else if self.pendant(vertex) {
                // This pendant vertex is getting a second edge added to it, and is no longer
                // pendant.
                self.set_pendant(vertex, false);
            }
        }

        // Then we merge the two components that the endpoints of the edge are in.
        match (
            self.component(edge.vertices[0]),
            self.component(edge.vertices[1]),
        ) {
            (None, None) => {
                // Use the smaller of the two indices of the edge's endpoints as the component
                // index. This is always `edge.vertices[0]`, since the vertices
                // of an edge are sorted by their index.
                let index = Self::component_index(edge.vertices[0]);
                for vertex in edge.vertices {
                    self.set_component(vertex, index)
                }
            }
            (Some(comp), None) => self.set_component(edge.vertices[1], comp),
            (None, Some(index)) => {
                // Check if `edge.vertices[0]` has the new smallest index in the component, and
                // if so update the component's index.
                let potential_index = Self::component_index(edge.vertices[0]);
                if potential_index < index {
                    self.set_component(edge.vertices[0], potential_index);
                    self.change_component_index(index, potential_index);
                } else {
                    self.set_component(edge.vertices[0], index);
                }
            }
            (Some(comp1), Some(comp2)) => {
                // Merge the two components, picking the smaller of their two indices as the new
                // index.
                let new_index = NonZeroU8::min(comp1, comp2);
                let old_index = NonZeroU8::max(comp1, comp2);
                self.change_component_index(old_index, new_index);
            }
        }
    }

    /// This function should be called when a vertex is part of an edge for the
    /// last time (is leaving the frontier), and checks that it satisfies all
    /// our requirements now that it's never going to be touched again.
    //
    /// If those requirements aren't met, it returns `Err(ConstantNode::Zero)`,
    /// which effectively discards this possiblility.
    ///
    /// Then, it does some bookkeeping:
    /// * If this vertex was the one in its component with the minimal index, it
    ///   updates the component's index to be based on the new vertex with the
    ///   minimal index, since we want it to be entirely determined by what's in
    ///   the component now, not what was there previously.
    /// * It resets the data for the vertex to 0, since we no longer care about
    ///   it now that it's exited and want two `NodeState`s which have different
    ///   info about it to be considered equal.
    ///
    /// Finally, it returns whether the component the vertex was in exited too,
    /// since it happens to need to compute that anyway to update the component
    /// index.
    fn handle_vertex_exit(&mut self, vertex: u8) -> Result<bool, ConstantNode> {
        if (0..8).contains(&vertex) {
            // The first 8 vertices are the corners of the cuboid.
            // We require that corners have at least one edge coming out of them; in other
            // words, that they're part of a connected component with size > 1.
            if self.component(vertex).is_none() {
                return Err(ConstantNode::Zero);
            }
        } else {
            // Vertices that aren't corners are required to not be pendant vertices, since
            // the cut edge that the lone edge coming out of the vertex represents should
            // just be glued together if that's the case.
            if self.pendant(vertex) {
                return Err(ConstantNode::Zero);
            }
        }

        let mut component_exited = false;

        // If this vertex was the one that determined the component's index, update it
        // to the new minimal index in the component.
        if self.component(vertex) == Some(Self::component_index(vertex)) {
            let old_index = Self::component_index(vertex);
            let new_index = (vertex + 1..self.num_vertices)
                .find(|&other_vertex| self.component(other_vertex) == Some(old_index))
                .map(Self::component_index);
            if let Some(new_index) = new_index {
                self.change_component_index(old_index, new_index);
            } else {
                // This was the last vertex in the component.
                component_exited = true;
            }
        }

        // Clear the data about the vertex.
        self.data[usize::from(vertex)] = 0;

        Ok(component_exited)
    }

    /// When a connected component of size > 1 has exited, finishes a graph.
    ///
    /// Since the graph is only allowed to contain one component of size > 1,
    /// when one exits, the graph is now either finished or invalid. So, this
    /// will always either return `Err(ConstantNode::Zero)` or
    /// `Err(ConstantNode::One)`.
    ///
    /// It's invalid if either there are vertices that haven't exited yet which
    /// don't satisfy their degree constraints, or there's another connected
    /// component with size > 1.
    fn finish(
        &mut self,
        remaining_edges: &[Edge],
        frontier: impl IntoIterator<Item = u8>,
    ) -> Result<(), ConstantNode> {
        if frontier
            .into_iter()
            .any(|vertex| self.component(vertex).is_some())
        {
            // There isn't allowed to be more than one connected component with size > 1, so
            // this case is invalid.
            return Err(ConstantNode::Zero);
        }

        // The graph is now finished, since there aren't any more edges that would
        // expand this connected component, and creating any new connected components
        // by adding more edges is illegal.
        //
        // Before we return `ConstantNode::One` to signal that, we handle the exit of
        // all the vertices that haven't already exited, since we're cutting things off
        // early and those checks won't get performed when they normally would when the
        // vertices exit the frontier.
        //
        // Note that we can't just use the frontier for this because it's possible that
        // there are some vertices that still haven't yet entered the frontier, and we
        // need to check those too.
        let remaining_vertices: HashSet<u8> = remaining_edges
            .iter()
            .flat_map(|edge| edge.vertices)
            .collect();
        for vertex in remaining_vertices {
            self.handle_vertex_exit(vertex)?;
        }

        // Now we finish the graph.
        Err(ConstantNode::One)
    }

    /// Returns the node that the 0-edge of this node should point to, when
    /// `edge` is being added.
    fn zero_edge(
        &self,
        edge: Edge,
        remaining_edges: &[Edge],
        frontier: &[u8],
    ) -> Result<NodeState, ConstantNode> {
        let mut new_node = *self;

        // First handle the exits of the individual vertices.
        let mut comp_exits: u8 = 0;
        // Probably pointless micro-optimisation: handle the exits in reverse so that we
        // don't have to reshuffle the component index twice.
        for vertex in edge.vertices.into_iter().rev() {
            if !frontier.contains(&vertex) {
                let comp_exited = new_node.handle_vertex_exit(vertex)?;
                if comp_exited {
                    comp_exits += 1;
                }
            }
        }

        match comp_exits {
            0 => {}
            1 => new_node.finish(remaining_edges, frontier.iter().copied())?,
            // Two components just exited with those vertices. That means for sure that the graph
            // has finished with more than one connected component of size > 1, which is invalid.
            2.. => return Err(ConstantNode::Zero),
        }

        Ok(new_node)
    }

    /// Returns the node that the 1-edge of this node should point to, when
    /// `edge` is being added.
    fn one_edge(
        &self,
        edge: Edge,
        remaining_edges: &[Edge],
        frontier: &[u8],
    ) -> Result<NodeState, ConstantNode> {
        if self.component(edge.vertices[0]).is_some()
            && self.component(edge.vertices[0]) == self.component(edge.vertices[1])
        {
            // Adding this edge would create a cycle, which isn't allowed, so
            // discard this possibility by pointing to the zero-edge.
            return Err(ConstantNode::Zero);
        }

        let mut new_node = *self;

        // Actually update the info in `new_node` to account for this edge being added.
        new_node.add_edge(edge);

        // Then we handle exiting vertices.
        let mut comp_exited = false;
        // Probably pointless micro-optimisation: handle the exits in reverse so that we
        // don't have to reshuffle the component index twice.
        for vertex in edge.vertices.into_iter().rev() {
            if !frontier.contains(&vertex) {
                // We don't have to worry about this happening more than once because we know
                // that the two vertices are in the same component, since we just connected
                // them.
                comp_exited |= new_node.handle_vertex_exit(vertex)?;
            }
        }

        // If a component exited, finish the graph.
        if comp_exited {
            new_node.finish(remaining_edges, frontier.iter().copied())?;
        }

        Ok(new_node)
    }
}

/// Entrypoint for a thread that computes one of the rows of a ZDD.
fn row_thread(
    edges: &[Edge],
    frontier: &[u8],
    progress_bars: &[ProgressBar],
    index: usize,
    receiver: mpsc::Receiver<NodeState>,
    sender: Option<mpsc::Sender<NodeState>>,
) -> Vec<Node> {
    let edge = edges[index];
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
                        .send(new_node)
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
        for v in 0..num_vertices {
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
                .map(|_| mpsc::channel::<NodeState>())
                .unzip();
            // Extract the first sender, which is used to send the initial `NodeState` to
            // the thread handling the first row.
            let mut senders = senders.into_iter();
            let first_sender = senders.next().unwrap();
            first_sender.send(NodeState::new(num_vertices)).unwrap();
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
                let frontier = &frontiers[index];
                let progress_bars = &progress_bars;
                let handle = thread::Builder::new()
                    .name(format!("row {}", index + 1))
                    .spawn_scoped(s, move || {
                        row_thread(edges, frontier, progress_bars, index, receiver, sender)
                    })
                    .unwrap();
                handles.push_back(handle);
            }

            rows.extend(handles.into_iter().map(|handle| handle.join().unwrap()));
            rows
        });

        Self {
            rotations: rotations_for_cuboid(cuboid, &vertices, &edges),
            cuboid,
            edges,
            vertices,
            rows,
        }
    }
}
