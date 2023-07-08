//! A crate which finds nets for cubiods.

// This file contains infrastructure shared between `primary.rs` and `alt.rs`.

use std::array;
use std::collections::HashSet;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Cuboid {
    pub width: u8,
    pub depth: u8,
    pub height: u8,
}

impl FromStr for Cuboid {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let dims: Vec<u8> = s
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
    pub fn new(width: u8, depth: u8, height: u8) -> Self {
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
        let width: usize = self.width.into();
        let height: usize = self.height.into();
        let depth: usize = self.depth.into();
        2 * width * depth + 2 * depth * height + 2 * width * height
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
    fn unique_cursors(&self) -> Vec<CursorData> {
        let mut result = Vec::new();

        // Don't bother with east, south, and top because they're symmetrical with west,
        // north and bottom, and so won't reveal any new nets.
        let mut faces = vec![Bottom, West, North];

        // Get rid of any faces that are the same size, since any point on one can be
        // rotated to a point on the other.
        let mut i = 0;
        while i < faces.len() {
            let face = faces[i];
            let face_size = self.face_size(face);
            if faces
                .iter()
                .filter(|&&other_face| other_face != face)
                .any(|&other_face| {
                    self.face_size(other_face) == face_size
                        || self.face_size(other_face) == face_size.rotated(1)
                })
            {
                faces.remove(i);
            } else {
                i += 1;
            }
        }

        for face in faces {
            let size = self.face_size(face);
            for x in 0..size.width {
                for y in 0..size.height {
                    // We only need to include orientations of 0 and 1 here, since 2 and 3 can be
                    // transformed into other spots on the same face with orientations of 0 and 1
                    // respectively with a 180 degree turn.
                    result.push(CursorData {
                        square: SquareData {
                            cuboid: *self,
                            face,
                            pos: Pos::new(x, y),
                        },
                        orientation: 0,
                    });
                    // If the cuboid is square, we don't even have to include an orientation of 1,
                    // since a 90 degree turn will transform it into another position on the same
                    // face with an orientation of 0.
                    // For non-square faces, this doesn't work because you end up with a completely
                    // differently shaped face after a 90 degree turn; a 180 degree turn is needed
                    // to get back to the same shape.
                    if size.width != size.height {
                        result.push(CursorData {
                            square: SquareData {
                                cuboid: *self,
                                face,
                                pos: Pos::new(x, y),
                            },
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
    width: u8,
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
        let (width, squares): (u8, Vec<bool>) = u.arbitrary()?;
        if width == 0 || squares.is_empty() {
            return Err(arbitrary::Error::IncorrectFormat);
        }
        Ok(Self { width, squares })
    }
}

impl Net {
    /// Creates a new, empty net with the given width and height.
    pub fn new(width: u8, height: u8) -> Self {
        Self {
            width,
            squares: vec![false; usize::from(width) * usize::from(height)],
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
        Self::new(width.try_into().unwrap(), height.try_into().unwrap())
    }

    pub fn width(&self) -> u8 {
        self.width
    }

    pub fn height(&self) -> u8 {
        if self.width == 0 {
            0
        } else {
            (self.squares.len() / usize::from(self.width))
                .try_into()
                .unwrap()
        }
    }

    fn size(&self) -> Size {
        Size::new(self.width(), self.height())
    }

    /// Returns an iterator over the rows of the net.
    pub fn rows(&self) -> impl Iterator<Item = &[bool]> + DoubleEndedIterator + ExactSizeIterator {
        self.squares.chunks(self.width.into())
    }

    /// Returns a mutable iterator over the rows of the net.
    pub fn rows_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut [bool]> + DoubleEndedIterator + ExactSizeIterator {
        self.squares.chunks_mut(self.width.into())
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
                    .unwrap_or(self.width().into())
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

        let mut result = Net::new(
            (end_x - start_x).try_into().unwrap(),
            (end_y - start_y).try_into().unwrap(),
        );

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
        self.squares[usize::from(pos.y) * usize::from(self.width) + usize::from(pos.x)]
    }

    /// Set whether a spot on the net is filled.
    fn set(&mut self, pos: Pos, value: bool) {
        self.squares[usize::from(pos.y) * usize::from(self.width) + usize::from(pos.x)] = value;
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
        for (a, b) in zip(
            start.chunks_mut(self.width.into()),
            end.rchunks_mut(self.width.into()),
        ) {
            // ignore the middle half-sized chunks if present
            if a.len() == usize::from(self.width) {
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
        let square_cache = SquareCache::new(cuboid);
        self.color_with_cache(cuboid, &square_cache)
    }

    /// Return a version of this net with its squares 'colored' with which faces
    /// they're on.
    ///
    /// This version of the method takes a `SquareCache` so it doesn't need to
    /// be re-computed every time.
    pub fn color_with_cache(
        &self,
        cuboid: Cuboid,
        square_cache: &SquareCache,
    ) -> Option<ColoredNet> {
        if self.area() != cuboid.surface_area() {
            return None;
        }

        // Find a filled-in square on the net, which we then assume maps to each spot on
        // the surface of the cuboid in turn, and try to find what face each spot on the
        // net would have to be if that was the case.
        // If we don't run into any contradictions, we've found a valid colouring.
        let index = self.squares.iter().position(|&square| square).unwrap();
        let net_start = Pos {
            x: (index % usize::from(self.width)).try_into().unwrap(),
            y: (index / usize::from(self.width)).try_into().unwrap(),
        };

        // Create a mapping from net positions to face positions.
        // A lot of time is spent on repeatedly allocating/deallocating this, so share
        // it between attempts.
        let mut pos_map = vec![None; self.squares.len()];

        for cursor in cuboid.unique_cursors() {
            let cursor = Cursor::from_data(&square_cache, &cursor);
            if self.try_color(cuboid, &square_cache, net_start, cursor, &mut pos_map) {
                return Some(ColoredNet {
                    width: self.width,
                    // Created a colored net from the position map by just taking which face each
                    // position is on.
                    squares: pos_map
                        .into_iter()
                        .map(|maybe_cursor| {
                            maybe_cursor.map(|cursor| cursor.to_data(square_cache).square.face)
                        })
                        .collect(),
                });
            }
        }

        None
    }

    fn try_color(
        &self,
        cuboid: Cuboid,
        square_cache: &SquareCache,
        net_start: Pos,
        surface_start: Cursor,
        pos_map: &mut [Option<Cursor>],
    ) -> bool {
        pos_map.fill(None);
        // Make a bitmask of what spots are filled.
        let mut surface: u64 = 0;

        fn fill(
            square_cache: &SquareCache,
            net: &Net,
            net_pos: Pos,
            surface_pos: Cursor,
            surface: &mut u64,
            pos_map: &mut [Option<Cursor>],
        ) -> bool {
            // Make a little helper function to get the index in `pos_map` which corresponds
            // to a given position.
            let index = |pos: Pos| usize::from(pos.y) * usize::from(net.width) + usize::from(pos.x);
            *surface |= 1 << surface_pos.square().0;
            pos_map[index(net_pos)] = Some(surface_pos);
            for direction in [Left, Up, Right, Down] {
                let Some(net_pos) = net_pos.moved_in(direction, net.size()) else {
                    continue;
                };

                if !net.filled(net_pos) {
                    // Skip this if the square isn't present in the original net.
                    continue;
                }

                let surface_pos = surface_pos.moved_in(square_cache, direction);

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

                if *surface & (1 << surface_pos.square().0) != 0 {
                    // This isn't a valid net for this cuboid (from this starting position) - this
                    // spot is on the net but the corresponding spot on the surface is filled, so
                    // the net doubles up which is invalid.
                    return false;
                }

                // Now try filling in this new position as well.
                if !fill(square_cache, net, net_pos, surface_pos, surface, pos_map) {
                    return false;
                }
            }

            true
        }

        fill(
            square_cache,
            self,
            net_start,
            surface_start,
            &mut surface,
            pos_map,
        ) && pos_map.iter().filter(|pos| pos.is_some()).count() == cuboid.surface_area()
    }

    pub fn area(&self) -> usize {
        self.squares.iter().filter(|&&square| square).count()
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
    width: u8,
    squares: Vec<Option<Face>>,
}

impl ColoredNet {
    pub fn new(width: u8, height: u8) -> Self {
        Self {
            width,
            squares: vec![None; usize::from(width) * usize::from(height)],
        }
    }

    /// Set whether a spot on the net is filled.
    pub fn set(&mut self, pos: Pos, value: Face) {
        self.squares[usize::from(pos.y) * usize::from(self.width) + usize::from(pos.x)] =
            Some(value);
    }

    pub fn get(&mut self, pos: Pos) -> Option<Face> {
        self.squares[usize::from(pos.y) * usize::from(self.width) + usize::from(pos.x)]
    }

    /// Returns an iterator over the rows of the net.
    fn rows(&self) -> impl Iterator<Item = &[Option<Face>]> {
        (self.width != 0)
            .then(|| self.squares.chunks(self.width.into()))
            .into_iter()
            .flatten()
    }

    pub fn area(&self) -> usize {
        self.squares
            .iter()
            .filter(|&&square| square.is_some())
            .count()
    }

    pub fn width(&self) -> u8 {
        self.width
    }

    pub fn height(&self) -> usize {
        if self.width == 0 {
            0
        } else {
            self.squares.len() / usize::from(self.width)
        }
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
    x: u8,
    y: u8,
}

impl Pos {
    pub fn new(x: u8, y: u8) -> Self {
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
    fn rotate(&mut self, turns: i8, size: Size) {
        match turns & 0b11 {
            0 => {}
            1 => (self.x, self.y) = (self.y, size.width - self.x - 1),
            2 => (self.x, self.y) = (size.width - self.x - 1, size.height - self.y - 1),
            3 => (self.x, self.y) = (size.height - self.y - 1, self.x),
            _ => unreachable!(),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Size {
    width: u8,
    height: u8,
}

impl Size {
    fn new(width: u8, height: u8) -> Self {
        Self { width, height }
    }

    /// Rotate the size by `turns` turns. Clockwise or anticlockwise, it
    /// produces the same result.
    fn rotate(&mut self, turns: i8) {
        if turns % 2 != 0 {
            std::mem::swap(&mut self.width, &mut self.height);
        }
    }

    fn rotated(mut self, turns: i8) -> Size {
        self.rotate(turns);
        self
    }

    fn is_square(&self) -> bool {
        self.width == self.height
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

/// A square on the surface of a cuboid.
///
/// You should pretty much always use `Square` (along with a `SquareCache`)
/// instead of this type, since it's smaller and has quicker lookup of
/// neighbouring squares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct SquareData {
    /// The cuboid that this square is on.
    cuboid: Cuboid,
    /// Which face this position is on.
    face: Face,
    /// The position within the face.
    pos: Pos,
}

impl SquareData {
    /// Creates a square representing (0, 0) on the bottom face of a cuboid.
    fn new(cuboid: Cuboid) -> Self {
        Self {
            cuboid,
            face: Bottom,
            pos: Pos::new(0, 0),
        }
    }

    /// Moves this square over by one in `direction`, assuming this square has
    /// the default orientation for its face.
    ///
    /// Note that if the square changes face, its implicit orientation isn't
    /// preserved, since that can change between faces. You need `CursorData`
    /// for that.
    fn move_in(&mut self, direction: Direction) {
        let success = self
            .pos
            .move_in(direction, self.cuboid.face_size(self.face));
        if !success {
            // We went off the edge of the face, time to switch faces.
            let (new_face, turns) = self.face.adjacent_in(direction);
            let entry_direction = direction.turned(turns);

            // Rotate the position to be aligned with the new face.
            self.pos.rotate(turns, self.cuboid.face_size(self.face));
            // Then set the coordinate we moved along to the appropriate edge of the face.
            match entry_direction {
                Left => self.pos.x = self.cuboid.face_size(new_face).width - 1,
                Up => self.pos.y = 0,
                Right => self.pos.x = 0,
                Down => self.pos.y = self.cuboid.face_size(new_face).height - 1,
            }

            self.face = new_face;
        }
    }

    /// Returns the square adjacent to this square in `direction`, assuming this
    /// square has the default orientation for its face.
    ///
    /// Note that if the square changes face, its implicit orientation isn't
    /// preserved, since that can change between faces. You need `CursorData`
    /// for that.
    fn moved_in(mut self, direction: Direction) -> Self {
        self.move_in(direction);
        self
    }
}

/// A cursor onto the surface of a cuboid.
///
/// A cursor lies on top of a square on the surface of a cuboid, and points in a
/// direction. I tend to imagine this as a square with a little chevron arrow
/// inside it.
///
/// The direction the cursor points in indicates which direction is up relative
/// to the cursor, so that you can then specify directions in terms of left, up,
/// down and right relative to the cursor.
///
/// To encode the direction, every face has a default cursor direction, and
/// `orientation` is the number of right turns away from that direction this
/// cursor is.
///
/// The main purpose of cursors are for keeping track of where squares move to
/// when you fold/unfold a cuboid. This includes both where squares end up on
/// the net after unfolding, and where they end up after unfolding into a net
/// and folding back up into a different cuboid.
///
/// Examples of usage:
/// - `Mapping` holds a cursor onto two different cuboids, and indicates that
///   with the net that's being built, when you unfold the first cuboid into
///   that net and then fold it back up into the second cuboid the first cursor
///   gets moved to the second cursor.
/// - `Instruction` holds a net position and a cursor onto each cuboid. The net
///   position is an implicit cursor onto the net pointing upwards, and then
///   each cursor on the cuboids is the position that the cursor on the net
///   moves to when you fold the net up into each cuboid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CursorData {
    /// The square the cursor is on.
    square: SquareData,
    /// The orientation of the cursor.
    orientation: i8,
}

impl CursorData {
    /// Creates a new curosr at (0, 0) on the bottom face of a cuboid, facing
    /// upwards.
    fn new(cuboid: Cuboid) -> Self {
        Self {
            square: SquareData::new(cuboid),
            orientation: 0,
        }
    }

    /// Returns the cuboid that this cursor is on.
    fn cuboid(&self) -> Cuboid {
        self.square.cuboid
    }

    /// Moves the cursor 1 unit in the given direction.
    fn move_in(&mut self, direction: Direction) {
        let face_direction = direction.turned(self.orientation);
        let success = self
            .square
            .pos
            .move_in(face_direction, self.cuboid().face_size(self.square.face));
        if !success {
            // We went off the edge of the face, time to switch faces.
            let (new_face, turns) = self.square.face.adjacent_in(face_direction);
            let entry_direction = face_direction.turned(turns);

            // Rotate the position to be aligned with the new face.
            self.square
                .pos
                .rotate(turns, self.cuboid().face_size(self.square.face));
            // Then set the coordinate we moved along to the appropriate edge of the face.
            match entry_direction {
                Left => self.square.pos.x = self.cuboid().face_size(new_face).width - 1,
                Up => self.square.pos.y = 0,
                Right => self.square.pos.x = 0,
                Down => self.square.pos.y = self.cuboid().face_size(new_face).height - 1,
            }

            self.orientation = (self.orientation + turns) & 0b11;
            self.square.face = new_face;
        }
    }

    /// Returns this cursor moved 1 unit in the given direction.
    fn moved_in(mut self, direction: Direction) -> Self {
        self.move_in(direction);
        self
    }

    /// Returns the list of cursors that are 'equivalent' to this one.
    ///
    /// This is the opposite of `unique_cursors`: it returns all of the cursors
    /// which can be reached by rotating the cuboid that this cursor lies on
    /// into an indistinguishable position while leaving the cursor where it is.
    fn equivalents(&self) -> Vec<CursorData> {
        let mut result = vec![*self];
        // If there are any other faces which are the same size, it's possible to rotate
        // the cuboid so that that face replaces this cursor's face and we get a cursor
        // on that face. So add an equivalent cursor on each so that everything we do
        // next gets applied to those faces too.
        for face in [Bottom, West, North]
            .into_iter()
            .filter(|&face| face != self.square.face && face != self.square.face.opposite())
            .filter(|&face| {
                self.cuboid().face_size(face) == self.cuboid().face_size(self.square.face)
                    || self.cuboid().face_size(face)
                        == self.cuboid().face_size(self.square.face).rotated(1)
            })
        {
            // Rotate our position to be correct on the new face.
            let mut cursor = *self;
            let (_, turns) = self.square.face.direction_of(face).unwrap();
            cursor
                .square
                .pos
                .rotate(turns, self.cuboid().face_size(self.square.face));
            cursor.orientation = (cursor.orientation + turns) & 0b11;
            cursor.square.face = face;
            result.push(cursor);
        }

        // Then we add an equivalent cursor on the opposite face, that you can get to by
        // rotating the cuboid 180 degrees.
        //
        // Funnily enough, you can actually get one by simply using the existing
        // coordinates on the new face - the way we've laid things out, everything has a
        // path to its opposite face which doesn't involve having to go through any
        // rotations, and so following that path around by 180 degrees leaves you with
        // the same position and orientation on the new face.
        for i in 0..result.len() {
            let mut cursor = result[i];
            cursor.square.face = cursor.square.face.opposite();
            result.push(cursor);
        }

        // Then add the cursors you get by rotating the face 180 degrees.
        for i in 0..result.len() {
            let mut cursor = result[i];
            cursor
                .square
                .pos
                .rotate(2, cursor.cuboid().face_size(cursor.square.face));
            cursor.orientation = (cursor.orientation + 2) & 0b11;
            result.push(cursor);
        }

        // Then, if the face is square, add the cursors you get by rotating the face 90
        // degrees (including from the 180-degree-rotated one we just made to get a
        // 270-degree version).
        if self.cuboid().face_size(self.square.face).is_square() {
            for i in 0..result.len() {
                let mut cursor = result[i];
                cursor
                    .square
                    .pos
                    .rotate(1, cursor.cuboid().face_size(cursor.square.face));
                cursor.orientation = (cursor.orientation + 1) & 0b11;
                result.push(cursor);
            }
        }

        result
    }
}

/// A cache of all the squares on the surface of a cuboid, along with a lookup
/// table for which square you reach by moving in any direction from another
/// square.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SquareCache {
    /// All the squares on the surface of a cuboid.
    ///
    /// This allows us to represent a square as an index in this array instead
    /// of having to copy it around everywhere, and also then reuse that index
    /// in a lookup table.
    squares: Vec<SquareData>,
    /// A lookup table which stores the face position you get by moving in every
    /// possible direction from a square, assuming the default orientation for
    /// its face.
    ///
    /// If `index` is the index of a square in `squares` and `direction` is the
    /// direction you want to find a face position adjacent in, the index you
    /// need in this array is `index << 2 | (direction as usize)`.
    neighbour_lookup: Vec<Cursor>,
}

impl SquareCache {
    /// Returns a `SquareCache` for the given cuboid.
    pub fn new(cuboid: Cuboid) -> SquareCache {
        assert!(
            cuboid.surface_area() <= 64,
            "`SquareCache` only supports up to 64 squares"
        );

        let mut squares = Vec::new();
        for face in [Bottom, West, North, East, South, Top] {
            for x in 0..cuboid.face_size(face).width {
                for y in 0..cuboid.face_size(face).height {
                    squares.push(SquareData {
                        cuboid,
                        face,
                        pos: Pos { x, y },
                    })
                }
            }
        }

        let neighbour_lookup = squares
            .iter()
            .flat_map(|&square| {
                [Left, Up, Right, Down].into_iter().map(move |direction| {
                    CursorData {
                        square,
                        orientation: 0,
                    }
                    .moved_in(direction)
                })
            })
            .map(|cursor| {
                let square = squares
                    .iter()
                    .position(|&square| square == cursor.square)
                    .unwrap()
                    .try_into()
                    .unwrap();
                Cursor::new(Square(square), cursor.orientation)
            })
            .collect();

        Self {
            squares,
            neighbour_lookup,
        }
    }

    /// Returns an iterator over all the squares in this `SquareCache`.
    fn squares(&self) -> impl Iterator<Item = Square> + DoubleEndedIterator + ExactSizeIterator {
        let num_squares: u8 = self
            .squares
            .len()
            .try_into()
            .ok()
            .filter(|&squares| squares < 64)
            .expect("`SquareCache` contained more than 64 squares");
        (0..num_squares).map(Square)
    }
}

/// A square on the surface of a cuboid, represented by an index into
/// `SquareCache::squares`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Square(u8);

impl Square {
    /// Returns a square representing (0, 0) on the bottom face of a cuboid.
    fn new() -> Self {
        Self(0)
    }

    /// Returns the square adjacent to this square in `direction`, assuming this
    /// square has the default orientation for its face.
    ///
    /// Note that if the square changes face, its implicit orientation isn't
    /// preserved, since that can change between faces. You need `Cursor` for
    /// that.
    fn moved_in(self, cache: &SquareCache, direction: Direction) -> Self {
        cache.neighbour_lookup[usize::from(self.0 << 2) | direction as usize].square()
    }

    /// Creates a `Square` from a `SquareData`.
    fn from_data(cache: &SquareCache, square: &SquareData) -> Self {
        let index = cache
            .squares
            .iter()
            .position(|other_square| other_square == square)
            .unwrap()
            .try_into()
            .unwrap();
        Self(index)
    }

    /// Converts a `Square` into a `SquareData`.
    fn to_data(self, cache: &SquareCache) -> SquareData {
        cache.squares[usize::from(self.0)]
    }
}

/// A cursor, bitwise-encoded as a square and an orientation. The square is
/// encoded in the upper 6 bits, and the orientation is encoded in the lower 2
/// bits.
///
/// A cursor lies on top of a square on the surface of a cuboid, and points in a
/// direction. I tend to imagine this as a square with a little chevron arrow
/// inside it.
///
/// The direction the cursor points in indicates which direction is up relative
/// to the cursor, so that you can then specify directions in terms of left, up,
/// down and right relative to the cursor.
///
/// To encode the direction, every face has a default cursor direction, and the
/// orientation is the number of right turns away from that direction this
/// cursor is pointing.
///
/// The main purpose of cursors are for keeping track of where squares move to
/// when you fold/unfold a cuboid. This includes both where squares end up on
/// the net after unfolding, and where they end up after unfolding into a net
/// and folding back up into a different cuboid.
///
/// Examples of usage:
/// - `Mapping` holds a cursor onto two different cuboids, and indicates that
///   with the net that's being built, when you unfold the first cuboid into
///   that net and then fold it back up into the second cuboid the first cursor
///   gets moved to the second cursor.
/// - `Instruction` holds a net position and a cursor onto each cuboid. The net
///   position is an implicit cursor onto the net pointing upwards, and then
///   each cursor on the cuboids is the position that the cursor on the net
///   moves to when you fold the net up into each cuboid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Cursor(u8);

impl Cursor {
    /// Builds a cursor from a square and an orientation.
    fn new(square: Square, orientation: i8) -> Self {
        assert!(square.0 < 64);
        Self(square.0 << 2 | (orientation & 0b11) as u8)
    }

    /// Returns the square where this position is.
    fn square(self) -> Square {
        Square(self.0 >> 2)
    }

    /// Returns the orientation of this position.
    fn orientation(self) -> i8 {
        (self.0 & 0b11) as i8
    }

    /// Returns this cursor moved over by one in `direction`.
    fn moved_in(self, cache: &SquareCache, direction: Direction) -> Self {
        let cursor = cache.neighbour_lookup
            [usize::from(self.0 & 0b11111100) | direction.turned(self.orientation()) as usize];
        // The neighbour lookup table only contains what you get when you move
        // in some direction from a particular _square_, which means our
        // orientation's been ignored; to fix that we just add it back on
        // afterwards.
        Cursor::new(cursor.square(), cursor.orientation() + self.orientation())
    }

    fn from_data(cache: &SquareCache, data: &CursorData) -> Self {
        let square = Square::from_data(cache, &data.square);
        Self(square.0 << 2 | (data.orientation & 0b11) as u8)
    }

    /// Converts a `Cursor` into a `CursorData`.
    fn to_data(self, cache: &SquareCache) -> CursorData {
        let square = self.square().to_data(cache);
        CursorData {
            square,
            orientation: self.orientation(),
        }
    }
}

/// A mapping between the surfaces of two cuboids, implemented as a pair of
/// cursors.
///
/// This can also be used to indicate where a spot on the net maps to on each
/// cuboid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MappingData {
    cursors: [CursorData; 2],
}

impl MappingData {
    /// Creates a `MappingData` between two cursors.
    fn new(cursor1: CursorData, cursor2: CursorData) -> Self {
        Self {
            cursors: [cursor1, cursor2],
        }
    }

    /// Returns the list of mappings that are 'equivalent' to this one.
    ///
    /// In effect, a mapping that's equivalent to this one is any mapping which
    /// will lead to the exact same list of common nets when used as a starting
    /// point from `NetFinder` (possibly rotated/flipped, but still the same
    /// net).
    ///
    /// The way I imagine this is that you overlay the two cuboids on top of one
    /// another, with the two cursors touching. Then, you can either:
    /// - Rotate one cuboid into an indistinguishable position, leaving its
    ///   cursor where it is, or
    /// - Rotate or flip both cuboids at the same time around the cursors,
    ///   leaving the cursors where they are.
    ///
    /// After doing either of those things to the pair of cuboids, they're
    /// pretty obviously still going to unfold into all the same things, but
    /// the cursors are now sitting on different spots on the cuboids. Thus
    /// we've got a mapping which is equivalent.
    fn equivalents(&self) -> HashSet<MappingData> {
        // First we get the list of equivalent cursors on each cuboid.
        let equivalent_cursors = self.cursors.map(|cursor| cursor.equivalents());
        // Then we create a mapping for every combination of those equivalent
        // cursors. This is a `HashSet` this time around because it's possible
        // for flipping to result in the same mapping we started with, and we
        // don't want any duplicates.
        let mut result = HashSet::new();
        for cursor1 in equivalent_cursors[0].iter().copied() {
            for cursor2 in equivalent_cursors[1].iter().copied() {
                result.insert(MappingData {
                    cursors: [cursor1, cursor2],
                });
            }
        }

        // Then try flipping both cuboids to get more equivalent mappings.
        // We have to store them in this intermediate `Vec` since we can't mutate
        // `result` while iterating over it.
        let mut batch = Vec::new();

        // First try doing a horizontal flip around the cursors.
        for mapping in result.iter() {
            let mut mapping = *mapping;
            for cursor in mapping.cursors.iter_mut() {
                let face_size = cursor.cuboid().face_size(cursor.square.face);
                if cursor.orientation % 2 == 0 {
                    // If the cursor is pointing vertically relative to its face, that means that it
                    // agrees with what the face thinks is horizontal, and we can just flip the x
                    // coordinate around.
                    cursor.square.pos.x = face_size.width - cursor.square.pos.x - 1;
                } else {
                    // Otherwise, horizontal relative to the cursor is vertical relative to the
                    // face, so we need to flip the y coordinate instead.
                    cursor.square.pos.y = face_size.height - cursor.square.pos.y - 1;
                }
            }
            batch.push(mapping);
        }
        result.extend(batch.drain(..));

        // Then a vertical flip.
        for mapping in result.iter() {
            let mut mapping = *mapping;
            for cursor in mapping.cursors.iter_mut() {
                let face_size = cursor.cuboid().face_size(cursor.square.face);
                // This logic is the same as for horizontal flips but in reverse.
                if cursor.orientation % 2 == 0 {
                    cursor.square.pos.y = face_size.height - cursor.square.pos.y - 1;
                } else {
                    cursor.square.pos.x = face_size.width - cursor.square.pos.x - 1;
                }
            }
            batch.push(mapping);
        }
        result.extend(batch.drain(..));

        // Then try rotating both cuboids to get more equivalent mappings. This just
        // amounts to simultaneously incrementing both cursors' orientations.
        for mapping in result.iter() {
            for turns in 1..=3 {
                let mut mapping = *mapping;
                for cursor in mapping.cursors.iter_mut() {
                    // The reason we have to AND this with `0b11` is so that the mapping
                    // de-duplication works properly: we need cursors with orientations of 0 and
                    // orientations of 4 to be considered the same (by translating them both to 0).
                    cursor.orientation = (cursor.orientation + turns) & 0b11;
                }
                batch.push(mapping);
            }
        }
        result.extend(batch.drain(..));

        result
    }
}

/// A mapping between the surfaces of two cuboids, implemented as a pair of
/// cursors.
///
/// This can also be used to indicate where a spot on the net maps to on each
/// cuboid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
// Aligning this to two bytes lets the compiler bitwise-encode it as a `u16`, making things faster.
#[repr(align(2))]
struct Mapping {
    cursors: [Cursor; 2],
}

impl Mapping {
    /// Creates a `Mapping` between two `Cursor`s.
    fn new(cursor1: Cursor, cursor2: Cursor) -> Self {
        Self {
            cursors: [cursor1, cursor2],
        }
    }

    /// Creates a `Mapping` from a `MappingData`, given the square caches for
    /// the two cursors in order.
    fn from_data(caches: &[SquareCache; 2], data: &MappingData) -> Self {
        Self {
            cursors: array::from_fn(|i| Cursor::from_data(&caches[i], &data.cursors[i])),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::hash::Hash;

    use crate::{
        Cuboid, Cursor, CursorData, Direction, Face, MappingData, Pos, Square, SquareCache,
        SquareData,
    };
    use Direction::*;
    use Face::*;

    #[test]
    fn square_cache() {
        let cuboid = Cuboid::new(1, 1, 5);
        let cache = SquareCache::new(cuboid);
        for square_data in cache.squares.iter().copied() {
            let square = Square::from_data(&cache, &square_data);
            assert_eq!(cache.squares[usize::from(square.0)], square_data);
            for direction in [Left, Up, Down, Right] {
                assert_eq!(
                    cache.squares[usize::from(square.moved_in(&cache, direction).0)],
                    square_data.moved_in(direction)
                );
                assert_eq!(
                    square.moved_in(&cache, direction),
                    Square::from_data(&cache, &square_data.moved_in(direction))
                );
            }

            for orientation in -4..=4 {
                let cursor = Cursor::new(square, orientation);
                assert_eq!(cursor.square(), square);
                assert_eq!(cursor.orientation(), orientation & 0b11);
                let cursor_data = CursorData {
                    square: square_data,
                    orientation,
                };
                for direction in [Left, Up, Down, Right] {
                    assert_eq!(
                        cursor.moved_in(&cache, direction),
                        Cursor::from_data(&cache, &cursor_data.moved_in(direction))
                    );
                }
            }
        }
    }

    #[test]
    fn equivalents() {
        fn cursor(cuboid: Cuboid, face: Face, x: u8, y: u8, orientation: i8) -> CursorData {
            CursorData {
                square: SquareData {
                    cuboid,
                    face,
                    pos: Pos { x, y },
                },
                orientation,
            }
        }

        fn hash_set<T: Eq + Hash>(iter: impl IntoIterator<Item = T>) -> HashSet<T> {
            iter.into_iter().collect()
        }

        let cuboid1 = Cuboid::new(1, 1, 5);
        let cursor1 = cursor(cuboid1, Bottom, 0, 0, 0);
        assert_eq!(
            hash_set(cursor1.equivalents()),
            hash_set([
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid1, Top, 0, 0, 3),
            ])
        );

        let cuboid2 = Cuboid::new(1, 2, 3);
        let cursor2 = cursor(cuboid2, West, 0, 0, 1);
        assert_eq!(
            hash_set(cursor2.equivalents()),
            hash_set([
                cursor(cuboid2, West, 0, 0, 1),
                cursor(cuboid2, West, 1, 2, 3),
                cursor(cuboid2, East, 0, 0, 1),
                cursor(cuboid2, East, 1, 2, 3),
            ])
        );

        let cursor3 = cursor(cuboid1, North, 0, 1, 2);
        assert_eq!(
            hash_set(cursor3.equivalents()),
            hash_set([
                cursor(cuboid1, North, 0, 1, 2),
                cursor(cuboid1, North, 0, 3, 0),
                cursor(cuboid1, East, 0, 1, 2),
                cursor(cuboid1, East, 0, 3, 0),
                cursor(cuboid1, South, 0, 1, 2),
                cursor(cuboid1, South, 0, 3, 0),
                cursor(cuboid1, West, 0, 1, 2),
                cursor(cuboid1, West, 0, 3, 0),
            ])
        );

        let mapping1 = MappingData::new(cursor1, cursor2);
        let expected_equivalents = hash_set([
            // First, there are all the direction combinations of equivalents of `cursor1` and
            // `cursor2`.
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 1, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 0, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 1, 2, 3),
            ),
            // Then, by performing a vertical flip you get a slightly different version of each of
            // those combinations.
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 0, 2, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 1, 0, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 0, 2, 3),
            ),
            // And _then_ there are the versions of all of those that you get by rotating both
            // cuboids at once.
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 0, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 1, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 1, 0, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 0, 2, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 0, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 1, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 1, 0, 3),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 0, 2, 1),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 0, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 1, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, West, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 3),
                cursor(cuboid2, East, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, West, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 0),
                cursor(cuboid2, East, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, West, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 1),
                cursor(cuboid2, East, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, West, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Bottom, 0, 0, 2),
                cursor(cuboid2, East, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, West, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 3),
                cursor(cuboid2, East, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, West, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 0),
                cursor(cuboid2, East, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, West, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 1),
                cursor(cuboid2, East, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, West, 0, 2, 2),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 1, 0, 0),
            ),
            MappingData::new(
                cursor(cuboid1, Top, 0, 0, 2),
                cursor(cuboid2, East, 0, 2, 2),
            ),
        ]);
        assert_eq!(mapping1.equivalents(), expected_equivalents)
    }
}
