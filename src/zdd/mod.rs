// This implements the ZDD-based algorithm for finding common developments of cuboids described in http://dx.doi.org/10.1016/j.comgeo.2017.03.001.

use std::{collections::VecDeque, hash::Hash, mem};

use crate::{Cuboid, Direction, Face, FacePos, Net, Pos};

use rayon::{
    iter::plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer},
    prelude::ParallelIterator,
};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use Direction::*;
use Face::*;

mod construct;

/// The max. number of vertices allowed.
/// This is 36 because that's all you need to be able to find ZDDs for 1x1x7 and
/// 1x2x5 cuboids.
/// (Oops, turns out I accidentally included support for 1x2x5 cuboids; 1x3x3
/// cuboids only have 32 vertices and 60 edges the same as 1x1x7 cuboids... oh
/// well)
const MAX_VERTICES: u8 = 36;
const MAX_EDGES: usize = 72;

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
#[repr(packed)]
struct Node {
    /// The combination of constant nodes and references to the next row that
    /// this node contains.
    state: u8,
    /// If the 0-edge of this node points to a node in the next row, the index
    /// of that node.
    zero_edge_index: u32,
    /// If the 1-edge of this node points to a node in the next row, the index
    /// of that node.
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
pub(self) struct Edge {
    /// The indices of the two endpoints of this edge.
    ///
    /// Sorted from smallest to largest.
    vertices: [u8; 2],
}

impl Edge {
    fn new(v1: u8, v2: u8) -> Self {
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

impl Zdd {
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
                let edge = Edge::new(i.try_into().unwrap(), j.try_into().unwrap());
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

    pub fn vertex_indices(&self) -> FxHashMap<(Face, Pos), u8> {
        let mut vertex_indices = FxHashMap::default();
        for (i, vertex) in self.vertices.iter().copied().enumerate() {
            for face in [Bottom, West, North, East, South, Top] {
                if let Some(pos) = vertex.pos_on_face(face) {
                    vertex_indices.insert((face, pos), i.try_into().unwrap());
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
    vertex_indices: &FxHashMap<(Face, Pos), u8>,
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
