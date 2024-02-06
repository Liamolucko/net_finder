use std::collections::{HashMap, HashSet};

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{Cuboid, Direction, Face, Pos, Transform, Zdd};

use Direction::*;
use Face::*;

/// A vertex of a cuboid.
///
/// Unlike in `Net` and `Surface`, this isn't the middle of a square on the
/// surface of a cuboid; it's the corner.
/// To be clear, it's not just the corners of the entire cuboid, it still
/// includes the corners of all the grid squares on the cuboid's surface.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(super) struct Vertex {
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
pub(super) struct Edge {
    /// The indices of the two endpoints of this edge.
    ///
    /// Sorted from smallest to largest.
    pub(super) vertices: [u8; 2],
}

impl Edge {
    pub(super) fn new(v1: u8, v2: u8) -> Self {
        let mut vertices = [v1, v2];
        vertices.sort_unstable();
        Self { vertices }
    }
}

impl Vertex {
    /// Returns the position of this vertex on the given face, if it's on that
    /// face.
    fn pos_on_face(&self, face: Face) -> Option<Pos> {
        if face == self.face {
            return Some(self.pos);
        }

        // Figure out which direction we have to go in to get onto that face, and if we
        // can even do so.
        let (direction, turns) = self.face.direction_of(face)?;
        let size = self.cuboid.face_size(self.face);
        match direction {
            Left if self.pos.x != 0 => return None,
            Up if self.pos.y != size.height => return None,
            Right if self.pos.x != size.width => return None,
            Down if self.pos.y != 0 => return None,
            _ => {}
        }

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
        let size = self.cuboid.face_size(face);
        match entry_direction {
            Left => new_pos.x = size.width,
            Up => new_pos.y = 0,
            Right => new_pos.x = 0,
            Down => new_pos.y = size.height,
        }

        Some(new_pos)
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
    pub(super) fn compute_geometry(cuboid: Cuboid) -> (Vec<Vertex>, Vec<Edge>) {
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

    /// Returns whether a set of edges encoded as a u64 is the canonical version
    /// (lexicographically smallest).
    pub(super) fn is_canon(&self, bits: u64) -> bool {
        self.rotations(bits).all(|rotated| bits <= rotated)
    }

    /// Returns all the rotations of a set of edges encoded as a u64.
    pub(super) fn rotations(&self, bits: u64) -> impl Iterator<Item = u64> + '_ {
        self.rotations
            .iter()
            .map(move |rotation| rotation.apply(bits))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct Rotation {
    /// A series of mask + rotate left operations that should be applied.
    ///
    /// For each one of these (mask, rotation) pairs, you apply that mask to the
    /// set of edges, rotate it left by the rotation, and then OR it into the
    /// result.
    ops: Vec<(u64, u8)>,
}

impl Rotation {
    pub(super) fn apply(&self, edges: u64) -> u64 {
        let mut result = 0;
        for &(mask, rotation) in self.ops.iter() {
            result |= (edges & mask).rotate_left(rotation.into())
        }
        result
    }
}

/// A 'work-in-progress' rotation, represented in the more intuitive format
/// of a mapping from one vertex to another, which is later converted to a
/// finalised `Rotation`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct WipRotation {
    mapping: Vec<u8>,
}

impl WipRotation {
    /// Creates a new `WipRotation` which just represents doing nothing.
    fn null(num_vertices: u8) -> Self {
        Self {
            mapping: (0..num_vertices).collect(),
        }
    }

    /// Creates a new `WipRotation` which represents flipping `face`
    /// horizontally, and taking the rest of the cuboid with it.
    fn horizontal_flip(vertices: &[Vertex], face: Face) -> Self {
        // Figure out what each vertex maps to when flipped.
        let mapping = vertices
            .iter()
            .map(|&vertex| {
                let face_size = vertex.cuboid.face_size(vertex.face);
                match face.direction_of(vertex.face) {
                    None => {
                        debug_assert!(vertex.face == face || vertex.face == face.opposite());
                        // The face onto which the horizontal flip is actually applied just
                        // stays where it is and has the vertices within it flipped.
                        Vertex {
                            cuboid: vertex.cuboid,
                            face: vertex.face,
                            pos: Pos {
                                x: face_size.width - vertex.pos.x,
                                y: vertex.pos.y,
                            },
                        }
                    }
                    Some((Up | Down, turns)) => {
                        // The faces above and below the flipped face also stay in place and get
                        // flipped, except that they get flipped vertically instead if they're
                        // off by 90 degrees.
                        if turns % 2 == 0 {
                            Vertex {
                                cuboid: vertex.cuboid,
                                face: vertex.face,
                                pos: Pos {
                                    x: face_size.width - vertex.pos.x,
                                    y: vertex.pos.y,
                                },
                            }
                        } else {
                            Vertex {
                                cuboid: vertex.cuboid,
                                face: vertex.face,
                                pos: Pos {
                                    x: vertex.pos.x,
                                    y: face_size.height - vertex.pos.y,
                                },
                            }
                        }
                    }
                    Some((Left | Right, _)) => {
                        // The faces to the left and right of the flipped face get switched with
                        // each other. The problem is, they aren't necessarily oriented the same
                        // way.
                        // Happily, it turns out that there are only two cases here though: each
                        // face is always either a vertical or horizontal flip different from its
                        // opposite face.
                        match vertex.face.opposite_transform() {
                            Transform::HorizontalFlip => Vertex {
                                cuboid: vertex.cuboid,
                                face: vertex.face.opposite(),
                                pos: Pos {
                                    x: face_size.width - vertex.pos.x,
                                    y: vertex.pos.y,
                                },
                            },
                            Transform::VerticalFlip => Vertex {
                                cuboid: vertex.cuboid,
                                face: vertex.face.opposite(),
                                pos: Pos {
                                    x: vertex.pos.x,
                                    y: face_size.height - vertex.pos.y,
                                },
                            },
                        }
                    }
                }
            })
            .map(|vertex| {
                vertices
                    .iter()
                    .position(|&other_vertex| other_vertex == vertex)
                    .unwrap()
                    .try_into()
                    .unwrap()
            })
            .collect();

        Self { mapping }
    }

    /// Creates a new `WipRotation` which represents flipping `face`
    /// vertically, and taking the rest of the cuboid with it.
    fn vertical_flip(vertices: &[Vertex], face: Face) -> Self {
        // Figure out what each vertex maps to when flipped.
        let mapping = vertices
            .iter()
            .map(|&vertex| {
                let face_size = vertex.cuboid.face_size(vertex.face);
                match face.direction_of(vertex.face) {
                    None => {
                        debug_assert!(vertex.face == face || vertex.face == face.opposite());
                        // The face onto which the vertical flip is actually applied just
                        // stays where it is and has the vertices within it flipped.
                        Vertex {
                            cuboid: vertex.cuboid,
                            face: vertex.face,
                            pos: Pos {
                                x: vertex.pos.x,
                                y: face_size.height - vertex.pos.y,
                            },
                        }
                    }
                    Some((Left | Right, turns)) => {
                        // The faces to the left and right of the flipped face also stay in place
                        // and get flipped, except that they get flipped horizontally instead if
                        // they're off by 90 degrees.
                        if turns % 2 == 0 {
                            Vertex {
                                cuboid: vertex.cuboid,
                                face: vertex.face,
                                pos: Pos {
                                    x: vertex.pos.x,
                                    y: face_size.height - vertex.pos.y,
                                },
                            }
                        } else {
                            Vertex {
                                cuboid: vertex.cuboid,
                                face: vertex.face,
                                pos: Pos {
                                    x: face_size.width - vertex.pos.x,
                                    y: vertex.pos.y,
                                },
                            }
                        }
                    }
                    Some((Up | Down, _)) => {
                        // The faces above and below the flipped face get switched with each other.
                        // The problem is, they aren't necessarily oriented the same way.
                        // Happily, it turns out that there are only two cases here though: each
                        // face is always either a vertical or horizontal flip different from
                        // its opposite face.
                        match vertex.face.opposite_transform() {
                            Transform::HorizontalFlip => Vertex {
                                cuboid: vertex.cuboid,
                                face: vertex.face.opposite(),
                                pos: Pos {
                                    x: face_size.width - vertex.pos.x,
                                    y: vertex.pos.y,
                                },
                            },
                            Transform::VerticalFlip => Vertex {
                                cuboid: vertex.cuboid,
                                face: vertex.face.opposite(),
                                pos: Pos {
                                    x: vertex.pos.x,
                                    y: face_size.height - vertex.pos.y,
                                },
                            },
                        }
                    }
                }
            })
            .map(|vertex| {
                vertices
                    .iter()
                    .position(|&other_vertex| other_vertex == vertex)
                    .unwrap()
                    .try_into()
                    .unwrap()
            })
            .collect();

        Self { mapping }
    }

    /// Creates a new `WipRotation` which represents rotating `face` clockwise
    /// by 90 degrees, and taking the rest of the cuboid with it.
    fn right_turn(vertices: &[Vertex], face: Face) -> Self {
        // Figure out what each vertex maps to when rotated.
        let mapping = vertices
            .iter()
            .map(|&vertex| {
                let face_size = vertex.cuboid.face_size(vertex.face);
                match face.direction_of(vertex.face) {
                    None => {
                        debug_assert!(vertex.face == face || vertex.face == face.opposite());
                        // The face onto which the rotation is actually applied just stays where it
                        // is and has the vertices within it rotated.
                        if vertex.face == face {
                            // The face itself gets rotated clockwise.
                            Vertex {
                                cuboid: vertex.cuboid,
                                face: vertex.face,
                                pos: Pos {
                                    x: vertex.pos.y,
                                    y: face_size.height - vertex.pos.x,
                                },
                            }
                        } else {
                            // However, the opposite face gets rotated anticlockwise.
                            Vertex {
                                cuboid: vertex.cuboid,
                                face: vertex.face,
                                pos: Pos {
                                    x: face_size.width - vertex.pos.y,
                                    y: vertex.pos.x,
                                },
                            }
                        }
                    }
                    Some((direction, turns)) => {
                        // On all the other faces, each vertex gets moved to the face next to it.

                        // First, figure out what direction we're entering the vertex's face from,
                        // within the perspective of that face.
                        let entry_direction = direction.turned(turns);
                        // Then, the direction we exit the face from is a 90 degrees clockwise turn
                        // of that.
                        let exit_direction = entry_direction.turned(1);

                        // Figure out which face we're moving onto.
                        let (new_face, turns) = vertex.face.adjacent_in(exit_direction);

                        // Then actually move onto it.
                        let mut new_pos = vertex.pos;
                        let mut size = vertex.cuboid.face_size(vertex.face);
                        for _ in 0..turns.rem_euclid(4) {
                            size.rotate(1);
                            new_pos = Pos {
                                x: new_pos.y,
                                y: size.height - new_pos.x,
                            }
                        }

                        Vertex {
                            cuboid: vertex.cuboid,
                            face: new_face,
                            pos: new_pos,
                        }
                    }
                }
            })
            .map(|vertex| {
                vertices
                    .iter()
                    .position(|&other_vertex| other_vertex == vertex)
                    .unwrap()
                    .try_into()
                    .unwrap()
            })
            .collect();

        Self { mapping }
    }

    /// Returns a new `WipRotation` which represents applying `self`
    /// followed by `next`.
    fn then(&self, next: &Self) -> Self {
        let mapping = self
            .mapping
            .iter()
            .map(|&index| next.mapping[usize::from(index)])
            .collect();
        Self { mapping }
    }

    /// Convert this into a finalized `Rotation`.
    fn finalize(&self, edges: &[Edge]) -> Rotation {
        // Apply this rotation to all the edges.
        let transformed: Vec<Edge> = edges
            .iter()
            .copied()
            .map(|mut edge| {
                for vertex in edge.vertices.iter_mut() {
                    *vertex = self.mapping[usize::from(*vertex)];
                }
                edge.vertices.sort_unstable();
                edge
            })
            .collect();

        // Then, find what rotation has to be applied to each edge, and store it in a
        // mapping from rotations to which edges have that shift.
        let mut rotations: HashMap<u8, Vec<u8>> = HashMap::new();
        for (src, dst_edge) in transformed.into_iter().enumerate() {
            let dst = edges.iter().position(|&edge| edge == dst_edge).unwrap();

            let src = i8::try_from(src).unwrap();
            let dst = i8::try_from(dst).unwrap();
            // First we calculate the amount we want to shift left by.
            let shift = dst - src;
            // Then, if it's negative (i.e. we actually want to shift right), we add 64,
            // which means that we'll rotate by 64 - right_shift, which means that we
            // effectively rotate left by 64, bringing us back to where we started, and then
            // shift right to where we want to go.
            let rotation = if shift > 0 {
                shift as u8
            } else {
                (shift + 64) as u8
            };
            rotations
                .entry(rotation)
                .or_default()
                .push(src.try_into().unwrap());
        }

        // Then combine together rotations of the same size into ops.
        let ops = rotations
            .into_iter()
            .map(|(shift, indices)| {
                let mut mask = 0;
                for index in indices {
                    mask |= 1 << index;
                }
                (mask, shift)
            })
            .collect();

        Rotation { ops }
    }
}

#[test]
fn elementary_rotations() {
    let (vertices, edges) = Zdd::compute_geometry(Cuboid::new(1, 1, 5));

    assert_eq!(
        WipRotation::horizontal_flip(&vertices, North),
        WipRotation::horizontal_flip(&vertices, Bottom)
    );
    assert_eq!(
        WipRotation::horizontal_flip(&vertices, South),
        WipRotation::horizontal_flip(&vertices, Bottom)
    );
    assert_eq!(
        WipRotation::horizontal_flip(&vertices, Top),
        WipRotation::horizontal_flip(&vertices, Bottom)
    );

    assert_eq!(
        WipRotation::vertical_flip(&vertices, North),
        WipRotation::vertical_flip(&vertices, West)
    );
    assert_eq!(
        WipRotation::vertical_flip(&vertices, East),
        WipRotation::vertical_flip(&vertices, West)
    );
    assert_eq!(
        WipRotation::vertical_flip(&vertices, South),
        WipRotation::vertical_flip(&vertices, West)
    );

    assert_eq!(
        WipRotation::horizontal_flip(&vertices, West),
        WipRotation::vertical_flip(&vertices, Bottom)
    );
    assert_eq!(
        WipRotation::horizontal_flip(&vertices, East),
        WipRotation::vertical_flip(&vertices, Bottom)
    );
    assert_eq!(
        WipRotation::vertical_flip(&vertices, Top),
        WipRotation::vertical_flip(&vertices, Bottom)
    );

    WipRotation::horizontal_flip(&vertices, Bottom).finalize(&edges);
    WipRotation::vertical_flip(&vertices, West).finalize(&edges);
    WipRotation::vertical_flip(&vertices, Bottom).finalize(&edges);

    WipRotation::right_turn(&vertices, Bottom).finalize(&edges);
}

pub(super) fn rotations_for_cuboid(
    cuboid: Cuboid,
    vertices: &[Vertex],
    edges: &[Edge],
) -> Vec<Rotation> {
    let mut elementary_rotations = vec![
        // Any other flip can be re-written as one of these three flips, so they're all we need to
        // include. A 180 degree rotation of a face can also be expressed as a horizontal
        // then vertical flip of that face, so there's no need to include that either.
        WipRotation::horizontal_flip(vertices, Bottom),
        WipRotation::vertical_flip(vertices, West),
        WipRotation::vertical_flip(vertices, Bottom),
    ];
    // For any square faces, we can also add in 90 degree rotations.
    // 270 degree rotations can just be expressed as 90 degree rotations plus 180
    // degree rotations.
    for face in [Bottom, West, North] {
        let size = cuboid.face_size(face);
        if size.width == size.height {
            elementary_rotations.push(WipRotation::right_turn(vertices, face));
        }
    }

    // Now, enumerate all the combinations of these rotations by repeatedly applying
    // all of them until we don't get anything new.
    let mut rotations = HashSet::new();
    rotations.insert(WipRotation::null(vertices.len().try_into().unwrap()));

    // We can't mutate `rotations` while iterating over it so we need this cache to
    // store new rotations in while we're iterating, so that we can then add them to
    // `rotations` after we're done.
    let mut cache = Vec::new();
    loop {
        let prev_len = rotations.len();

        for elementary_rotation in elementary_rotations.iter() {
            for rotation in rotations.iter() {
                cache.push(rotation.then(elementary_rotation));
            }
        }
        rotations.extend(cache.drain(..));

        if rotations.len() == prev_len {
            // We couldn't create any new rotations, so we must have found them all.
            break;
        }
    }

    rotations
        .into_iter()
        .map(|rotation| rotation.finalize(edges))
        .collect()
}

#[test]
fn rotations() {
    let cuboid = Cuboid::new(1, 1, 5);
    let (vertices, edges) = Zdd::compute_geometry(cuboid);
    let rotations = rotations_for_cuboid(cuboid, &vertices, &edges);
    for rotation in rotations {
        assert_eq!(rotation.apply(0), 0);
        assert_eq!(rotation.apply(0x00000fffffffffff), 0x00000fffffffffff)
    }
}
