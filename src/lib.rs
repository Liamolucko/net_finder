//! A crate which finds nets for cubiods.

// This file contains infrastructure shared between `primary.rs` and `alt.rs`.

use std::fmt::{self, Display, Formatter, Write};
use std::hash::{Hash, Hasher};
use std::iter::zip;
use std::str::FromStr;

use anyhow::{bail, Context};
use arbitrary::Arbitrary;
use serde::{Deserialize, Serialize};

mod primary;
mod zdd;

pub use primary::*;
pub use zdd::*;

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
            .split('x')
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

    /// Returns all the face positions on the surface of a cuboid that can't be
    /// reached by taking what another face position becomes when you rotate the
    /// cuboid.
    ///
    /// This is useful because when picking spots on nets/surfaces to be
    /// equivalent, there's no point trying two different face positions that
    /// can be transformed into one another by rotating the cuboid, since then
    /// you just end up with the exact same solution net but with squares
    /// corresponding to different face positions.
    fn surface_squares(&self) -> Vec<FacePos> {
        let mut result = Vec::new();
        // Don't bother with east, south, and top because they're symmetrical with west,
        // north and bottom, and so won't reveal any new nets.
        for face in [Bottom, West, North] {
            let size = self.face_size(face);
            for x in 0..size.width {
                for y in 0..size.height {
                    // We only need to include orientations of 0 and 1 here, since 2 and 3 can be
                    // transformed into other spots on the same face with orientations of 0 and 1
                    // respectively with a 180 degree turn.
                    result.push(FacePos {
                        cuboid: *self,
                        face,
                        pos: Pos::new(x, y),
                        orientation: 0,
                    });
                    // If the cuboid is square, we don't even have to include an orientation of 1,
                    // since a 90 degree turn will transform it into another position on the same
                    // face with an orientation of 0.
                    // For non-square faces, this doesn't work because you end up with a completely
                    // differently shaped face after a 90 degree turn; a 180 degree turn is needed
                    // to get back to the same shape.
                    if size.width != size.height {
                        result.push(FacePos {
                            cuboid: *self,
                            face,
                            pos: Pos::new(x, y),
                            orientation: 1,
                        })
                    }
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
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Create the 'canonical' version of this net by finding the rotation which
        // results in the lexicographically 'largest' value of `squares`.
        let canon = self.canon();

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
        if width == 0 || squares.is_empty() {
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

    /// Returns a net big enough to fit any net for any of the passed cuboids.
    pub fn for_cuboids(cuboids: &[Cuboid]) -> Self {
        // We put our starting point (the bottom-left of the bottom face) in the middle
        // of the net, and then the net could potentially extend purely left, right,
        // up, down etc. from there. So, the x/y of that point is the maximum
        // width/height of the net once shrunk (minus one, because zero-indexing).
        let middle_x = cuboids
            .iter()
            .map(|cuboid| cuboid.surface_area() - 1)
            .max()
            .unwrap();
        let middle_y = cuboids
            .iter()
            .map(|cuboid| cuboid.surface_area() - 1)
            .max()
            .unwrap();
        // Then the max width/height is 1 for the middle, plus twice the maximum
        // extension off to either side, which happens to be equal to the middle's
        // index.
        let width = 1 + 2 * middle_x;
        let height = 1 + 2 * middle_y;
        Self::new(width, height)
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
    pub fn rows(&self) -> impl Iterator<Item = &[bool]> + DoubleEndedIterator + ExactSizeIterator {
        self.squares.chunks(self.width)
    }

    /// Returns a mutable iterator over the rows of the net.
    pub fn rows_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut [bool]> + DoubleEndedIterator + ExactSizeIterator {
        self.squares.chunks_mut(self.width)
    }

    /// Return a copy of this net with the empty space removed from around it.
    pub fn shrink(&self) -> Self {
        let start_y = self
            .rows()
            .position(|row| row.iter().map(|&square| square as usize).sum::<usize>() != 0);
        let Some(start_y) = start_y else {
            // The net contains no squares. Just return an empty net.
            return Net::new(0, 0);
        };
        let end_y = self
            .rows()
            .rposition(|row| row.iter().map(|&square| square as usize).sum::<usize>() != 0)
            .unwrap()
            + 1;

        let start_x = self
            .rows()
            .take(end_y)
            .skip(start_y)
            .map(|row| {
                row.iter()
                    .position(|&square| square)
                    .unwrap_or(self.width())
            })
            .min()
            .unwrap();
        let end_x = self
            .rows()
            .take(end_y)
            .skip(start_y)
            .map(|row| row.iter().rposition(|&square| square).unwrap_or(0) + 1)
            .max()
            .unwrap();

        let mut result = Net::new(end_x - start_x, end_y - start_y);

        for (src, dst) in zip(self.rows().take(end_y).skip(start_y), result.rows_mut()) {
            dst.copy_from_slice(&src[start_x..end_x])
        }

        result
    }

    pub fn canon(&self) -> Self {
        Rotations::new(self.shrink())
            .max_by(|a, b| a.squares.as_slice().cmp(b.squares.as_slice()))
            .unwrap()
    }

    /// Returns whether a given position on the net is filled.
    pub fn filled(&self, pos: Pos) -> bool {
        self.squares[pos.y * self.width + pos.x]
    }

    /// Set whether a spot on the net is filled.
    fn set(&mut self, pos: Pos, value: bool) {
        self.squares[pos.y * self.width + pos.x] = value;
    }

    /// Returns a copy of this net rotated by the given number of clockwise
    /// turns.
    pub fn rotate(&self, turns: i8) -> Self {
        let turns = turns & 0b11;
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

    /// Returns a copy of this net flipped around the vertical axis.
    pub fn vertically_flipped(&self) -> Self {
        let mut result = self.clone();
        result.vertical_flip();
        result
    }

    /// Return a version of this net with its squares 'colored' with which faces
    /// they're on.
    pub fn color(&self, cuboid: Cuboid) -> Option<ColoredNet> {
        if self.area() != cuboid.surface_area() {
            return None;
        }

        // Find a filled-in square on the net, which we then assume maps to each spot on
        // the surface of the cuboid in turn, and try to find what face each spot on the
        // net would have to be if that was the case.
        // If we don't run into any contradictions, we've found a valid colouring.
        let index = self.squares.iter().position(|&square| square).unwrap();
        let net_start = Pos {
            x: index % self.width,
            y: index / self.width,
        };

        // A lot of time is spent on repeatedly allocating/deallocating these, so share
        // them between attempts.
        let mut surface = Surface::new(cuboid);
        // Create a mapping from net positions to face positions.
        let mut pos_map = vec![None; self.squares.len()];

        for square in cuboid.surface_squares() {
            if self.try_color(cuboid, net_start, square, &mut surface, &mut pos_map) {
                return Some(ColoredNet {
                    width: self.width,
                    // Created a colored net from the position map by just taking which face each
                    // position is on.
                    squares: pos_map
                        .into_iter()
                        .map(|maybe_pos| maybe_pos.map(|pos| pos.face))
                        .collect(),
                });
            }
        }

        None
    }

    fn try_color(
        &self,
        cuboid: Cuboid,
        net_start: Pos,
        surface_start: FacePos,
        surface: &mut Surface,
        pos_map: &mut [Option<FacePos>],
    ) -> bool {
        surface.clear();
        pos_map.fill(None);

        fn fill(
            net: &Net,
            net_pos: Pos,
            surface_pos: FacePos,
            surface: &mut Surface,
            pos_map: &mut [Option<FacePos>],
        ) -> bool {
            // Make a little helper function to get the index in `pos_map` which corresponds
            // to a given position.
            let index = |pos: Pos| pos.y * net.width + pos.x;
            surface.set(surface_pos, true);
            pos_map[index(net_pos)] = Some(surface_pos);
            for direction in [Left, Up, Right, Down] {
                let Some(net_pos) = net_pos.moved_in(direction, net.size()) else {
                    continue;
                };

                if !net.filled(net_pos) {
                    // Skip this if the square isn't present in the original net.
                    continue;
                }

                let surface_pos = surface_pos.moved_in(direction);

                if let Some(existing_surface_pos) = pos_map[index(net_pos)] {
                    if existing_surface_pos == surface_pos {
                        // This square has already been covered, with the same result.
                        continue;
                    } else {
                        // We got a different face position for this square by coming in from a
                        // different direction, which is invalid.
                        return false;
                    }
                }

                if surface.filled(surface_pos) {
                    // This isn't a valid net for this cuboid (from this starting position) - this
                    // spot is on the net but the corresponding spot on the surface is filled, so
                    // the net doubles up which is invalid.
                    return false;
                }

                // Now try filling in this new position as well.
                if !fill(net, net_pos, surface_pos, surface, pos_map) {
                    return false;
                }
            }

            true
        }

        fill(self, net_start, surface_start, surface, pos_map)
            && pos_map.iter().filter(|pos| pos.is_some()).count() == cuboid.surface_area()
    }

    pub fn area(&self) -> usize {
        self.squares.iter().filter(|&&square| square).count()
    }

    fn clear(&mut self) {
        self.squares.fill(false);
    }
}

#[test]
fn color() {
    const F: bool = false;
    const T: bool = true;
    let net = Net {
        width: 10,
        #[rustfmt::skip]
        squares: vec![
            T, T, F, F, F, F, F, F, F, F,
            F, T, F, F, T, F, F, T, F, F,
            T, T, T, T, T, F, F, T, F, F,
            F, F, T, T, T, T, T, T, T, T,
            F, F, F, F, F, F, F, T, F, F,
            F, F, F, F, F, F, F, T, F, F,
            F, F, F, F, F, F, F, T, F, F,
        ],
    };
    let cuboids = [Cuboid::new(1, 1, 5), Cuboid::new(1, 2, 3)];
    for cuboid in cuboids {
        assert!(net.color(cuboid).is_some());
    }
}

/// A version of `Net` which stores which face each of its squares are on.
#[derive(Debug, Clone)]
pub struct ColoredNet {
    width: usize,
    squares: Vec<Option<Face>>,
}

impl ColoredNet {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            squares: vec![None; width * height],
        }
    }

    /// Set whether a spot on the net is filled.
    pub fn set(&mut self, pos: Pos, value: Face) {
        self.squares[pos.y * self.width + pos.x] = Some(value);
    }

    pub fn get(&mut self, pos: Pos) -> Option<Face> {
        self.squares[pos.y * self.width + pos.x]
    }

    /// Returns an iterator over the rows of the net.
    fn rows(&self) -> impl Iterator<Item = &[Option<Face>]> {
        (self.width != 0)
            .then(|| self.squares.chunks(self.width))
            .into_iter()
            .flatten()
    }

    /// Returns a mutable iterator over the rows of the net.
    fn rows_mut(&mut self) -> impl Iterator<Item = &mut [Option<Face>]> {
        (self.width != 0)
            .then(|| self.squares.chunks_mut(self.width))
            .into_iter()
            .flatten()
    }

    pub fn area(&self) -> usize {
        self.squares
            .iter()
            .filter(|&&square| square.is_some())
            .count()
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
                if cell.is_some() {
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
            return ColoredNet::new(0, 0);
        }

        let mut result = ColoredNet::new(end_x - start_x, end_y - start_y);

        for (src, dst) in zip(self.rows().take(end_y).skip(start_y), result.rows_mut()) {
            dst.copy_from_slice(&src[start_x..end_x])
        }

        result
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
pub struct Pos {
    x: usize,
    y: usize,
}

impl Pos {
    pub fn new(x: usize, y: usize) -> Self {
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
        for _ in 0..turns & 0b11 {
            (self.x, self.y) = (self.y, size.width - self.x - 1);
            size.rotate(1);
        }
    }

    fn rotated(mut self, turns: i8, size: Size) -> Self {
        self.rotate(turns, size);
        self
    }
}

impl Display for Pos {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
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
    fn turned(self, turns: i8) -> Self {
        match i8::wrapping_add(self as i8, turns) & 0b11 {
            0 => Left,
            1 => Up,
            2 => Right,
            3 => Down,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Face {
    Bottom,
    West,
    North,
    East,
    South,
    Top,
}

impl Face {
    /// Returns the face adjacent to this face in the given direction, as well
    /// as the amount of times you'll need to turn a position on the current
    /// face clockwise to translate it to the new face.
    fn adjacent_in(self, direction: Direction) -> (Face, i8) {
        let table = [
            // bottom
            (West, 1),
            (North, 0),
            (East, -1),
            (South, 2),
            // west
            (South, 0),
            (Top, 1),
            (North, 0),
            (Bottom, -1),
            // north
            (West, 0),
            (Top, 0),
            (East, 0),
            (Bottom, 0),
            // east
            (North, 0),
            (Top, -1),
            (South, 0),
            (Bottom, 1),
            // south
            (East, 0),
            (Top, 2),
            (West, 0),
            (Bottom, 2),
            // top
            (West, -1),
            (South, 2),
            (East, 1),
            (North, 0),
        ];
        table[(self as usize) << 2 | direction as usize]
    }

    fn direction_of(self, other: Face) -> Option<(Direction, i8)> {
        match (self, other) {
            // bottom
            (Bottom, West) => Some((Left, 1)),
            (Bottom, North) => Some((Up, 0)),
            (Bottom, East) => Some((Right, -1)),
            (Bottom, South) => Some((Down, 2)),
            // west
            (West, South) => Some((Left, 0)),
            (West, Top) => Some((Up, 1)),
            (West, North) => Some((Right, 0)),
            (West, Bottom) => Some((Down, -1)),
            // north
            (North, West) => Some((Left, 0)),
            (North, Top) => Some((Up, 0)),
            (North, East) => Some((Right, 0)),
            (North, Bottom) => Some((Down, 0)),
            // east
            (East, North) => Some((Left, 0)),
            (East, Top) => Some((Up, -1)),
            (East, South) => Some((Right, 0)),
            (East, Bottom) => Some((Down, 1)),
            // south
            (South, East) => Some((Left, 0)),
            (South, Top) => Some((Up, 2)),
            (South, West) => Some((Right, 0)),
            (South, Bottom) => Some((Down, 2)),
            // top
            (Top, West) => Some((Left, -1)),
            (Top, South) => Some((Up, 2)),
            (Top, East) => Some((Right, 1)),
            (Top, North) => Some((Down, 0)),

            _ => None,
        }
    }

    /// Returns the face opposite to this one.
    fn opposite(self) -> Face {
        match self {
            Bottom => Top,
            West => East,
            North => South,
            East => West,
            South => North,
            Top => Bottom,
        }
    }

    /// Returns the transformation necessary to convert from a position on this
    /// face to a position on the face opposite to it.
    fn opposite_transform(&self) -> Transform {
        match self {
            Bottom | Top => Transform::VerticalFlip,
            West | North | East | South => Transform::HorizontalFlip,
        }
    }
}

use Face::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Transform {
    HorizontalFlip,
    VerticalFlip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FacePos {
    /// The cuboid which this position is on.
    cuboid: Cuboid,
    /// Which face this position is on.
    pub face: Face,
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
            let (new_face, turns) = self.face.adjacent_in(face_direction);
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

            self.orientation = (self.orientation + turns) & 0b11;
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

    fn clear(&mut self) {
        for face in &mut self.faces {
            face.clear();
        }
    }
}

impl PartialEq for Surface {
    fn eq(&self, other: &Self) -> bool {
        zip(&self.faces, &other.faces).all(|(a, b)| a.squares == b.squares)
    }
}

impl Eq for Surface {}

impl Hash for Surface {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for face in &self.faces {
            face.squares.hash(state);
        }
    }
}
