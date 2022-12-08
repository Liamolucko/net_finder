//! A crate which finds nets for cubiods.

use std::collections::{HashSet, VecDeque};
use std::fmt::{self, Display, Formatter, Write};
use std::hash::Hash;
use std::iter::zip;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::mpsc;
use std::{fs, thread};

use anyhow::{bail, Context};
use arbitrary::Arbitrary;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Cuboid {
    pub width: usize,
    pub depth: usize,
    pub height: usize,
}

impl FromStr for Cuboid {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let dims: Vec<usize> = s
            .split("x")
            .map(|dim| dim.parse())
            .collect::<Result<_, _>>()
            .context("dimensions must be integers")?;
        let &[width, depth, height] = dims.as_slice() else {
            bail!("3 dimensions must be provided: width, depth and height")
        };
        Ok(Self {
            width,
            depth,
            height,
        })
    }
}

impl Display for Cuboid {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}x{}", self.width, self.depth, self.height)
    }
}

impl Cuboid {
    pub fn new(width: usize, depth: usize, height: usize) -> Self {
        Self {
            width,
            depth,
            height,
        }
    }

    /// Returns the size of the given face.
    fn face_size(&self, face: Face) -> Size {
        match face {
            Bottom | Top => Size::new(self.width, self.depth),
            West | East => Size::new(self.depth, self.height),
            North | South => Size::new(self.width, self.height),
        }
    }

    pub fn surface_area(&self) -> usize {
        2 * self.width * self.depth + 2 * self.depth * self.height + 2 * self.width * self.height
    }

    fn surface_squares(&self) -> Vec<FacePos> {
        let mut result = Vec::new();
        // Don't bother with east, south, and top because they're symmetrical with west,
        // north and bottom, and so won't reveal any new nets.
        for face in [Bottom, West, North] {
            let size = self.face_size(face);
            for x in 0..size.width {
                for y in 0..size.height {
                    // At first, I thought we would have to include all the different rotations
                    // here. However, because of cuboids' rotational symmetry, any rotation is
                    // equivalent to just putting the square somewhere else on the same face.
                    result.push(FacePos {
                        cuboid: *self,
                        face,
                        pos: Pos::new(x, y),
                        orientation: 0,
                    })
                }
            }
        }
        result
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Net {
    width: usize,
    /// A vec indicating whether a square is present in each spot.
    squares: Vec<bool>,
}

struct Rotations {
    net: Net,
    stage: u8,
}

impl Rotations {
    fn new(net: Net) -> Self {
        Self { net, stage: 0 }
    }
}

impl Iterator for Rotations {
    type Item = Net;

    fn next(&mut self) -> Option<Self::Item> {
        match self.stage {
            1 | 3 | 5 | 7 => self.net.horizontal_flip(),
            2 | 6 => self.net.vertical_flip(),
            0 => {}
            4 => self.net = self.net.rotate(1),
            _ => return None,
        }
        self.stage += 1;
        Some(self.net.clone())
    }
}

impl PartialEq for Net {
    fn eq(&self, other: &Self) -> bool {
        let a = self.shrink();
        let b = other.shrink();

        Rotations::new(b).any(|b| a.width == b.width && a.squares == b.squares)
    }
}

impl Eq for Net {}

impl Hash for Net {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Create the 'canonical' version of this net by finding the rotation which
        // results in the lexicographically 'largest' value of `squares`.
        let canon = Rotations::new(self.shrink())
            .max_by(|a, b| a.squares.as_slice().cmp(b.squares.as_slice()))
            .unwrap();

        canon.width.hash(state);
        canon.squares.hash(state);
    }
}

#[test]
fn rotated_cross() {
    let cross = Net {
        width: 3,
        #[rustfmt::skip]
        squares: vec![
            false, true, false,
            true, true, true,
            false, true, false,
            false, true, false,
        ],
    };
    assert_eq!(cross, cross.rotate(1));
}

impl<'a> Arbitrary<'a> for Net {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let (width, squares): (usize, Vec<bool>) = u.arbitrary()?;
        if width == 0 || squares.len() == 0 {
            return Err(arbitrary::Error::IncorrectFormat);
        }
        Ok(Self { width, squares })
    }
}

impl Net {
    /// Creates a new, empty net with the given width and height.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            squares: vec![false; width * height],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        if self.width == 0 {
            0
        } else {
            self.squares.len() / self.width
        }
    }

    fn size(&self) -> Size {
        Size::new(self.width(), self.height())
    }

    /// Returns an iterator over the rows of the net.
    pub fn rows(&self) -> impl Iterator<Item = &[bool]> {
        (self.width != 0)
            .then(|| self.squares.chunks(self.width))
            .into_iter()
            .flatten()
    }

    /// Returns a mutable iterator over the rows of the net.
    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [bool]> {
        (self.width != 0)
            .then(|| self.squares.chunks_mut(self.width))
            .into_iter()
            .flatten()
    }

    /// Return a copy of this net with the empty space removed from around it.
    pub fn shrink(&self) -> Self {
        // The x and y at which the new net will start (inclusive).
        // If these get left as their initial values after the end of the net, the net
        // contains no squares.
        let mut start_x = self.width();
        let mut start_y = self.height();
        // The x and y at which the new net will end (exclusive).
        let mut end_x = 0;
        let mut end_y = 0;

        for (y, row) in self.rows().enumerate() {
            for (x, &cell) in row.iter().enumerate() {
                if cell {
                    // There's a cell in this row, so push the end row to the next row.
                    end_y = y + 1;
                    if x >= end_x {
                        // There's a cell in this column, so push the end column to the next column.
                        end_x = x + 1;
                    }

                    if x < start_x {
                        start_x = x;
                    }
                    if y < start_y {
                        start_y = y;
                    }
                }
            }
        }

        if start_x == self.width() || start_y == self.height() {
            // The net contains no squares. Just return an empty net.
            return Net::new(0, 0);
        }

        let mut result = Net::new(end_x - start_x, end_y - start_y);

        for (src, dst) in zip(self.rows().take(end_y).skip(start_y), result.rows_mut()) {
            dst.copy_from_slice(&src[start_x..end_x])
        }

        result
    }

    /// Returns whether a given position on the net is filled.
    fn filled(&self, pos: Pos) -> bool {
        self.squares[pos.y * self.width + pos.x]
    }

    /// Set whether a spot on the net is filled.
    fn set(&mut self, pos: Pos, value: bool) {
        self.squares[pos.y * self.width + pos.x] = value;
    }

    /// Returns a copy of this net rotated by the given number of clockwise
    /// turns.
    pub fn rotate(&self, turns: i8) -> Self {
        let turns = turns.rem_euclid(4);
        if turns == 0 {
            return self.clone();
        }
        let mut rotated_size = self.size();
        rotated_size.rotate(turns);
        let mut result = Net::new(rotated_size.width, rotated_size.height);
        for x in 0..self.width() {
            for y in 0..self.height() {
                let pos = Pos::new(x, y);
                let rotated = pos.rotated(turns, self.size());
                result.set(rotated, self.filled(pos));
            }
        }
        result
    }

    /// Flips this net around the horizontal axis.
    pub fn horizontal_flip(&mut self) {
        for row in self.rows_mut() {
            row.reverse();
        }
    }

    /// Returns a copy of this net flipped around the horizontal axis.
    pub fn horizontally_flipped(&self) -> Self {
        let mut result = self.clone();
        result.horizontal_flip();
        result
    }

    pub fn vertical_flip(&mut self) {
        let midpoint = self.squares.len() / 2;
        let (start, end) = self.squares.split_at_mut(midpoint);
        for (a, b) in zip(start.chunks_mut(self.width), end.rchunks_mut(self.width)) {
            // ignore the middle half-sized chunks if present
            if a.len() == self.width {
                a.swap_with_slice(b);
            }
        }
    }

    /// Return a version of this net with its squares 'colored' with which faces
    /// they're on.
    pub fn color(&self, cuboid: Cuboid) -> Option<ColoredNet> {
        if self.area() != cuboid.surface_area() {
            return None;
        }
        for x in 0..self.width() {
            for y in 0..self.height() {
                let pos = Pos::new(x, y);
                if self.filled(pos) {
                    for orientation in 0..4 {
                        if let Some(net) = self.try_color(cuboid, pos, orientation) {
                            return Some(net);
                        }
                    }
                }
            }
        }
        None
    }

    fn try_color(&self, cuboid: Cuboid, start: Pos, orientation: i8) -> Option<ColoredNet> {
        let mut surface_start = FacePos::new(cuboid);
        surface_start.orientation = orientation;
        // We need to keep track of what we've filled in to reject cases where stuff
        // overlaps.
        let mut surface = Surface::new(cuboid);
        let mut result = ColoredNet {
            width: self.width,
            squares: vec![None; self.squares.len()],
        };
        // This queue isn't a weird one like in `NetFinder`, it's a normal queue where
        // we push things on and pop things off.
        let mut queue = VecDeque::new();
        queue.push_back((start, surface_start));

        while let Some((pos, surface_pos)) = queue.pop_front() {
            result.set(pos, surface_pos.face);
            surface.set(surface_pos, true);
            for direction in [Left, Up, Right, Down] {
                let Some(pos) = pos.moved_in(direction, self.size()) else {
                    continue
                };
                if !self.filled(pos) || result.get(pos).is_some() {
                    // skip this if the square isn't present or we've already covered it.
                    continue;
                }
                let surface_pos = surface_pos.moved_in(direction);
                if surface.filled(surface_pos) {
                    // This isn't a valid net for this cuboid (from this starting position) - this
                    // spot is on the net but the corresponding spot on the surface is filled, so
                    // the net doubles up which is invalid.
                    return None;
                }
                queue.push_back((pos, surface_pos));
            }
        }

        if result.area() == cuboid.surface_area() {
            Some(result)
        } else {
            None
        }
    }

    pub fn area(&self) -> usize {
        self.squares.iter().filter(|&&square| square).count()
    }
}

/// A version of `Net` which stores which face each of its squares are on.
pub struct ColoredNet {
    width: usize,
    squares: Vec<Option<Face>>,
}

impl ColoredNet {
    /// Set whether a spot on the net is filled.
    fn set(&mut self, pos: Pos, value: Face) {
        self.squares[pos.y * self.width + pos.x] = Some(value);
    }

    fn get(&mut self, pos: Pos) -> Option<Face> {
        self.squares[pos.y * self.width + pos.x]
    }

    /// Returns an iterator over the rows of the net.
    fn rows(&self) -> impl Iterator<Item = &[Option<Face>]> {
        (self.width != 0)
            .then(|| self.squares.chunks(self.width))
            .into_iter()
            .flatten()
    }

    pub fn area(&self) -> usize {
        self.squares
            .iter()
            .filter(|&&square| square.is_some())
            .count()
    }
}

impl Display for ColoredNet {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (i, row) in self.rows().enumerate() {
            if i != 0 {
                f.write_char('\n')?;
            }
            for &square in row {
                if let Some(face) = square {
                    f.write_str(match face {
                        Bottom => "\x1b[31m",
                        West => "\x1b[32m",
                        North => "\x1b[33m",
                        East => "\x1b[34m",
                        South => "\x1b[35m",
                        Top => "\x1b[36m",
                    })?;
                    f.write_char('█')?;
                } else {
                    f.write_char(' ')?;
                }
            }
            f.write_str("\x1b[39m")?;
        }
        Ok(())
    }
}

/// Displays a text version of the net.
impl Display for Net {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let shrunk = self.shrink();
        for (i, row) in shrunk.rows().enumerate() {
            if i != 0 {
                f.write_char('\n')?;
            }
            for &cell in row {
                if cell {
                    f.write_char('█')?;
                } else {
                    f.write_char(' ')?;
                }
            }
        }
        Ok(())
    }
}

// A position in 2D space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Pos {
    x: usize,
    y: usize,
}

impl Pos {
    fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }

    /// Move the position 1 unit in the given direction.
    ///
    /// Returns whether the movement was successful; `false` means that the
    /// position went off the edge of the net.
    ///
    /// If `false` is returned, `self` is unchanged.
    fn move_in(&mut self, direction: Direction, size: Size) -> bool {
        match direction {
            Left if self.x > 0 => self.x -= 1,
            Up if self.y < size.height - 1 => self.y += 1,
            Right if self.x < size.width - 1 => self.x += 1,
            Down if self.y > 0 => self.y -= 1,
            // One of our bounds checks failed.
            _ => return false,
        }
        true
    }

    /// Returns this position moved 1 unit in the given direction, if it fits
    /// within the given size.
    fn moved_in(mut self, direction: Direction, size: Size) -> Option<Self> {
        if self.move_in(direction, size) {
            Some(self)
        } else {
            None
        }
    }

    /// Returns what this position would be if the surface it was on was rotated
    /// by `turns` clockwise turns, and the surface was of size `size` before
    /// being turned.
    fn rotate(&mut self, turns: i8, mut size: Size) {
        for _ in 0..turns.rem_euclid(4) {
            (self.x, self.y) = (self.y, size.width - self.x - 1);
            size.rotate(1);
        }
    }

    fn rotated(mut self, turns: i8, size: Size) -> Self {
        self.rotate(turns, size);
        self
    }
}

// A size in 2D space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Size {
    width: usize,
    height: usize,
}

impl Size {
    fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
    /// Rotate the size by `turns` turns. Clockwise or anticlockwise, it
    /// produces the same result.
    fn rotate(&mut self, turns: i8) {
        if turns % 2 != 0 {
            std::mem::swap(&mut self.width, &mut self.height);
        }
    }
}

// A direction in 2D space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Direction {
    Left,
    Up,
    Right,
    Down,
}
use Direction::*;

impl Direction {
    /// Returns `self` rotated by a given number of clockwise turns.
    fn turned(mut self, turns: i8) -> Self {
        for _ in 0..turns.rem_euclid(4) {
            self = match self {
                Left => Up,
                Up => Right,
                Right => Down,
                Down => Left,
            }
        }
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
enum Face {
    Bottom,
    West,
    North,
    East,
    South,
    Top,
}

use Face::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct FacePos {
    /// The cuboid which this position is on.
    cuboid: Cuboid,
    /// Which face this position is on.
    face: Face,
    /// The position within the face.
    pos: Pos,
    /// The orientation of the current face.
    ///
    /// Formatted as the number of clockwise turns that must be made to a
    /// direction on the net to convert it to a direction on the current face
    orientation: i8,
}

impl FacePos {
    /// Creates a new face position at (0, 0) on the bottom face of the given
    /// cuboid, oriented the same as the net.
    fn new(cuboid: Cuboid) -> Self {
        Self {
            cuboid,
            face: Bottom,
            pos: Pos::new(0, 0),
            orientation: 0,
        }
    }

    /// Move the position 1 unit in the given direction on the net.
    fn move_in(&mut self, direction: Direction) {
        let face_direction = direction.turned(self.orientation);
        let success = self
            .pos
            .move_in(face_direction, self.cuboid.face_size(self.face));
        if !success {
            // We went off the edge of the face, time to switch faces.
            let (new_face, turns) = match (self.face, face_direction) {
                // bottom
                (Bottom, Left) => (West, 1),
                (Bottom, Up) => (North, 0),
                (Bottom, Right) => (East, -1),
                (Bottom, Down) => (South, 2),
                // west
                (West, Left) => (South, 0),
                (West, Up) => (Top, 1),
                (West, Right) => (North, 0),
                (West, Down) => (Bottom, -1),
                // north
                (North, Left) => (West, 0),
                (North, Up) => (Top, 0),
                (North, Right) => (East, 0),
                (North, Down) => (Bottom, 0),
                // east
                (East, Left) => (North, 0),
                (East, Up) => (Top, -1),
                (East, Right) => (South, 0),
                (East, Down) => (Bottom, 1),
                // south
                (South, Left) => (East, 0),
                (South, Up) => (Top, 2),
                (South, Right) => (West, 0),
                (South, Down) => (Bottom, 2),
                // top
                (Top, Left) => (West, -1),
                (Top, Up) => (South, 2),
                (Top, Right) => (East, 1),
                (Top, Down) => (North, 0),
            };
            let entry_direction = face_direction.turned(turns);

            // Rotate the position to be aligned with the new face.
            self.pos.rotate(turns, self.cuboid.face_size(self.face));
            // Then set the coordinate we moved along to the appropriate edge of the face.
            match entry_direction {
                Left => self.pos.x = self.cuboid.face_size(new_face).width - 1,
                Up => self.pos.y = 0,
                Right => self.pos.x = 0,
                Down => self.pos.y = self.cuboid.face_size(new_face).height - 1,
            }

            self.orientation = (self.orientation + turns).rem_euclid(4);
            self.face = new_face;
        }
    }

    /// Returns this position moved 1 unit in the given direction on the net.
    fn moved_in(mut self, direction: Direction) -> Self {
        self.move_in(direction);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Surface {
    faces: [Net; 6],
}

impl Surface {
    fn new(cuboid: Cuboid) -> Self {
        Self {
            faces: [
                Net::new(cuboid.width, cuboid.depth),
                Net::new(cuboid.depth, cuboid.height),
                Net::new(cuboid.width, cuboid.height),
                Net::new(cuboid.depth, cuboid.height),
                Net::new(cuboid.width, cuboid.height),
                Net::new(cuboid.width, cuboid.depth),
            ],
        }
    }

    fn filled(&self, pos: FacePos) -> bool {
        self.faces[pos.face as usize].filled(pos.pos)
    }

    fn set(&mut self, pos: FacePos, value: bool) {
        self.faces[pos.face as usize].set(pos.pos, value)
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NetFinder {
    net: Net,
    surfaces: Vec<Surface>,
    queue: Vec<Instruction>,
    /// The index of the next instruction in `queue` that will be evaluated.
    index: usize,
    /// The size of the net so far.
    area: usize,
    target_area: usize,
}

/// An instruction to add a square.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Instruction {
    /// The position on the net where the square will be added.
    net_pos: Pos,
    /// The equivalent positions on the surfaces of the cuboids.
    face_positions: Vec<FacePos>,
    /// If this instruction has been successfully carried out, the index in
    /// `queue` at which the instructions added as a result of this instruction
    /// begin.
    ///
    /// In other words, the length of the queue prior to this instruction being
    /// carried out.
    followup_index: Option<usize>,
}

impl PartialEq for Instruction {
    fn eq(&self, other: &Self) -> bool {
        self.net_pos == other.net_pos && self.face_positions == other.face_positions
    }
}

impl NetFinder {
    /// Create a bunch of `NetFinder`s that search for all the nets that fold
    /// into all of the passed cuboids.
    fn new(cuboids: &[Cuboid]) -> anyhow::Result<Vec<Self>> {
        if cuboids.len() == 0 {
            bail!("at least one cuboid must be provided")
        }
        if cuboids
            .iter()
            .any(|cuboid| cuboid.surface_area() != cuboids[0].surface_area())
        {
            bail!("all passed cuboids must have the same surface area")
        }

        // Allocate the biggest net we could possibly need. We return a shrunk version.
        // We put our starting point (the bottom-left of the bottom face) in the middle
        // of the net, and then the net could potentially extend purely left, right,
        // up, down etc. from there. So, the x/y of that point is the maximum
        // width/height of the net once shrunk (minus one, because zero-indexing).
        let middle_x = cuboids
            .iter()
            .map(|cuboid| 2 * cuboid.width + 2 * cuboid.height - 1)
            .max()
            .unwrap();
        let middle_y = cuboids
            .iter()
            .map(|cuboid| 2 * cuboid.depth + 2 * cuboid.height - 1)
            .max()
            .unwrap();
        // Then the max width/height is 1 for the middle, plus twice the maximum
        // extension off to either side, which happens to be equal to the middle's
        // index.
        let width = 1 + 2 * middle_x;
        let height = 1 + 2 * middle_y;
        let net = Net::new(width, height);

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
        starting_points[0] = vec![FacePos::new(cuboids[0])];

        // let mut starting_face_positions =
        // vec![cuboids.iter().copied().map(FacePos::new).collect();
        // starting_points.iter().map(|vec| vec.len()).product()];
        let mut starting_face_positions: Vec<Vec<FacePos>> = Vec::new();
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
                    followup_index: None,
                }];
                Self {
                    net: net.clone(),
                    surfaces: surfaces.clone(),
                    queue,
                    index: 0,
                    area: 0,
                    target_area: cuboids[0].surface_area(),
                }
            })
            .collect())
    }

    /// Set a square on the net and surfaces of the cuboids simultaneously.
    fn set_square(&mut self, net_pos: Pos, face_positions: &[FacePos], value: bool) {
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
            .rfind(|(_, instruction)| instruction.followup_index.is_some()) else {
                return false
            };
        let followup_index = instruction.followup_index.unwrap();
        // Mark it as unsuccessful.
        instruction.followup_index = None;
        // Remove the square it added.
        let net_pos = instruction.net_pos;
        // inefficient :(
        let face_positions = instruction.face_positions.clone();
        self.set_square(net_pos, &face_positions, false);
        // Then remove all the instructions added as a result of this square.
        self.queue.resize_with(followup_index, || unreachable!());
        // We continue executing from just after the instruction we undid.
        self.index = last_success_index + 1;
        true
    }

    /// Evaluate the instruction at the current index in the queue, incrementing
    /// the index afterwards.
    fn evaluate_instruction(&mut self) {
        let next_index = self.queue.len();
        let Instruction {
            net_pos,
            ref face_positions,
            ..
        } = self.queue[self.index];
        // this is inefficient but I'm too lazy to fight the borrow checker right now.
        let face_positions = face_positions.clone();
        if !self.net.filled(net_pos)
            && !zip(&self.surfaces, &face_positions).any(|(surface, &pos)| surface.filled(pos))
            && !self.queue.iter().any(|instruction| {
                // If we find an instruction which is trying to put a square in the same
                // position on the net, but will produce a square in a different position on one
                // of the cuboids' surfaces, that means a cut would be needed, and this square
                // is invalid.
                instruction.net_pos == net_pos && instruction.face_positions != face_positions
            })
        {
            // The space is free, so we fill it.
            self.set_square(net_pos, &face_positions, true);
            self.queue[self.index].followup_index = Some(next_index);

            // Add the new things we can do from here to the queue.
            let new_instructions: Vec<_> = [Left, Up, Right, Down]
                .into_iter()
                .filter_map(|direction| {
                    Some(Instruction {
                        net_pos: net_pos.moved_in(direction, self.net.size())?,
                        face_positions: face_positions
                            .iter()
                            .map(|&pos| pos.moved_in(direction))
                            .collect(),
                        followup_index: None,
                    })
                })
                .filter(|instruction| !self.queue.contains(instruction))
                .collect();
            self.queue.extend_from_slice(&new_instructions);
        }
        self.index += 1;
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

    Ok(run(cuboids, finders))
}

pub fn resume(cuboids: &[Cuboid]) -> anyhow::Result<impl Iterator<Item = Net>> {
    let bytes = fs::read(state_path(cuboids)).context("no state to resume from")?;
    let finders: Vec<NetFinder> = postcard::from_bytes(&bytes)?;
    Ok(run(cuboids, finders))
}

fn run(cuboids: &[Cuboid], finders: Vec<NetFinder>) -> impl Iterator<Item = Net> {
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
                    finder.evaluate_instruction();

                    send_counter = send_counter.wrapping_add(1);
                    if send_counter == 0 {
                        let _ = finder_tx.try_send(finder.clone());
                    }
                }

                // We broke out of the loop, which means we reached the target area and have a
                // valid net!
                let net = finder.net.shrink();
                finder.backtrack();

                if let Err(_) = net_tx.send(net) {
                    // The receiver is gone, i.e. the iterator was dropped.
                    return;
                }
            }
        });
    }

    println!("threads spawned!");

    let cuboids = cuboids.to_vec();
    // Spawn a thread whose sole purpose is to periodically record our state to a
    // file.
    thread::spawn(move || {
        // Create the folder where we're going to store our state.
        fs::create_dir_all(Path::new(env!("CARGO_MANIFEST_DIR")).join("state")).unwrap();
        loop {
            let finders: Vec<_> = finder_receivers
                .iter_mut()
                .filter_map(|rx| rx.recv().ok())
                .collect();

            fs::write(
                state_path(&cuboids),
                &postcard::to_stdvec(&finders).unwrap(),
            )
            .unwrap();
        }
    });

    let mut yielded_nets = HashSet::new();
    net_rx.into_iter().filter(move |net| {
        let new = yielded_nets.insert(net.clone());
        new
    })
}
