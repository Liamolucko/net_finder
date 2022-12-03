//! A crate which finds nets for cubiods.

use std::fmt::{self, Display, Formatter, Write};
use std::iter::zip;

use arbitrary::Arbitrary;

#[derive(Debug, Clone, Copy)]
pub struct Cuboid {
    pub width: usize,
    pub depth: usize,
    pub height: usize,
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
}

impl Display for Cuboid {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}x{} cuboid", self.width, self.depth, self.height)
    }
}

#[derive(Clone, Debug)]
pub struct Net {
    width: usize,
    /// A vec indicating whether a square is present in each spot.
    squares: Vec<bool>,
}

impl<'a> Arbitrary<'a> for Net {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let (width, squares): (usize, Vec<bool>) = u.arbitrary()?;
        if width == 0 {
            if squares.len() != 0 {
                return Err(arbitrary::Error::IncorrectFormat);
            }
        } else if squares.len() % width != 0 {
            return Err(arbitrary::Error::IncorrectFormat);
        } else if squares.len() == 0 && width != 0 {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    /// Returns what this position would be if the surface it was on was rotated
    /// by `turns` clockwise turns, and the surface was of size `size` before
    /// being turned.
    fn rotate(&mut self, turns: i8, mut size: Size) {
        for _ in 0..turns.rem_euclid(4) {
            (self.x, self.y) = (self.y, size.width - self.x - 1);
            size.rotate(1);
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy)]
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
                Up => self.pos.y = self.cuboid.face_size(new_face).height - 1,
                Right => self.pos.x = 0,
                Down => self.pos.y = 0,
            }

            self.orientation = (self.orientation + turns).rem_euclid(4);
            self.face = new_face;
        }
    }
}

/// Finds a net which can fold into any of the passed set of cuboids.
pub fn find_nets(cuboid: Cuboid) -> Net {
    // We start from the bottom-left of the bottom face of the cuboid, and work our
    // way out from there. The net is aligned with the bottom face, so left, right,
    // up, down on the net correspond to the same directions on the bottom face.
    // From there they continue up other faces, e.g. left on the net can become up
    // on the left face.
    // We think of directions on each face as they appear from the inside.

    // Allocate the biggest net we could possibly need. We return a shrunk version
    // from the iterator.
    // We put our starting point (the bottom-left of the bottom face) in the middle
    // of the net, and then the net could potentially extend purely left, right,
    // up, down etc. from there. So, the x/y of that point is the maximum
    // width/height of the net once shrunk (minus one, because zero-indexing).
    let middle_x = 2 * cuboid.width + 2 * cuboid.height - 1;
    let middle_y = 2 * cuboid.depth + 2 * cuboid.height - 1;
    // Then the max width/height is 1 for the middle, plus twice the maximum
    // extension off to either side, which happens to be equal to the middle's
    // index.
    let width = 1 + 2 * middle_x;
    let height = 1 + 2 * middle_y;
    let mut net = Net::new(width, height);

    // The state of how much each face of the cuboid has been filled in. I reused
    // `Net` to represent the state of each face.
    // I put them in the arbitrary order of bottom, west, north, east, south, top.
    let mut faces = [
        Net::new(cuboid.width, cuboid.depth),
        Net::new(cuboid.depth, cuboid.height),
        Net::new(cuboid.width, cuboid.height),
        Net::new(cuboid.depth, cuboid.height),
        Net::new(cuboid.width, cuboid.height),
        Net::new(cuboid.width, cuboid.depth),
    ];

    // Construct the initial positions.
    let face_pos = FacePos::new(cuboid);
    let net_pos = Pos::new(middle_x, middle_y);

    // Set the first square.
    net.set(net_pos, true);
    faces[face_pos.face as usize].set(face_pos.pos, true);

    // A list of all the squares we're going to attempt to fill.
    // Formatted as `(pos, direction, success)`, where each element means 'extrude a
    // square in `direction` from `pos`', and `success` is whether it was
    // actually carried out.
    // However, we don't remove anything from the queue, because we need to be able
    // to backtrack.
    let mut queue: Vec<_> = [Left, Up, Right, Down]
        .into_iter()
        .map(|direction| (net_pos, face_pos, direction, false))
        .collect();
    // The index of the next instruction in the queue that we're going to carry out.
    let mut index = 0;

    // Compute the target surface area.
    // Once we reach it, we're done.
    let target_area = cuboid.surface_area();
    let mut area = 1;

    while area < target_area {
        if index >= queue.len() {
            // We've exhausted the queue and haven't found a solution, so we
            // must have done something wrong.
            // Backtrack and undo the last instruction we successfully carried
            // out.
            let (last_success_index, &mut (mut net_pos, mut face_pos, direction, ref mut success)) =
                queue
                    .iter_mut()
                    .enumerate()
                    .rfind(|(_, &mut (_, _, _, success))| success)
                    .unwrap();
            *success = false;
            // Figure out which square was added, and remove it.
            let moved = net_pos.move_in(direction, net.size());
            debug_assert!(moved);
            face_pos.move_in(direction);
            net.set(net_pos, false);
            faces[face_pos.face as usize].set(face_pos.pos, false);
            area -= 1;
            // Then remove all the instructions added as a result of this square.
            let (delete_start_index, _) = queue
                .iter()
                .enumerate()
                .find(|(_, &(other_net_pos, _, _, _))| other_net_pos == net_pos)
                .unwrap();
            queue.resize_with(delete_start_index, || unreachable!());
            index = last_success_index + 1;
            // it's possible that that last successful one was actually the last one in the
            // queue, so restart to loop to run this if again and check for that.
            continue;
        }

        let (mut net_pos, mut face_pos, direction, ref mut success) = queue[index];
        let moved = net_pos.move_in(direction, net.size());
        // we should always have allocated a big enough net.
        debug_assert!(moved);
        if !net.filled(net_pos) {
            face_pos.move_in(direction);
            if !faces[face_pos.face as usize].filled(face_pos.pos) {
                // Success!
                *success = true;
                net.set(net_pos, true);
                faces[face_pos.face as usize].set(face_pos.pos, true);
                area += 1;

                // Add the new things we can do from here to the queue.
                queue.extend(
                    [Left, Up, Right, Down]
                        .into_iter()
                        .map(|direction| (net_pos, face_pos, direction, false)),
                )
            }
        }
        index += 1;
    }

    net.shrink()
}
