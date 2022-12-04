//! A crate which finds nets for cubiods.

use std::fmt::{self, Display, Formatter, Write};
use std::iter::zip;

use anyhow::bail;
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

impl PartialEq for Net {
    fn eq(&self, other: &Self) -> bool {
        let a = self.shrink();
        let mut b = other.shrink();
        let mut rotated = b.rotate(1);

        fn sub_eq(a: &Net, b: &Net) -> bool {
            a.width == b.width && a.squares == b.squares
        }

        if sub_eq(&a, &b) {
            return true;
        }

        b.horizontal_flip();

        if sub_eq(&a, &b) {
            return true;
        }

        // this results in an overall rotation of 180 degrees.
        b.vertical_flip();

        if sub_eq(&a, &b) {
            return true;
        }

        b.horizontal_flip();

        if sub_eq(&a, &b) {
            return true;
        }

        if sub_eq(&a, &rotated) {
            return true;
        }

        rotated.horizontal_flip();

        if sub_eq(&a, &rotated) {
            return true;
        }

        rotated.vertical_flip();

        if sub_eq(&a, &rotated) {
            return true;
        }

        rotated.horizontal_flip();

        if sub_eq(&a, &rotated) {
            return true;
        }

        false
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

struct Surface {
    faces: [Net; 6],
}

impl Surface {
    fn filled(&self, pos: FacePos) -> bool {
        self.faces[pos.face as usize].filled(pos.pos)
    }

    fn set(&mut self, pos: FacePos, value: bool) {
        self.faces[pos.face as usize].set(pos.pos, value)
    }
}

pub struct NetFinder {
    net: Net,
    surfaces: Vec<Surface>,
    queue: Vec<Instruction>,
    /// The index of the next instruction in `queue` that will be evaluated.
    index: usize,
    /// The size of the net so far.
    area: usize,
    target_area: usize,
    /// All of the nets we've previously reached, to stop us from wasting effort
    /// if we reach them again.
    prev_nets: Vec<Net>,
}

/// An instruction to add a square.
#[derive(Debug, Clone)]
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

impl NetFinder {
    /// Create a `NetFinder` that searches for nets that fold into all of the
    /// passed cuboids.
    pub fn new(cuboids: Vec<Cuboid>) -> anyhow::Result<Self> {
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
        let surfaces = cuboids
            .iter()
            .map(|cuboid| Surface {
                faces: [
                    Net::new(cuboid.width, cuboid.depth),
                    Net::new(cuboid.depth, cuboid.height),
                    Net::new(cuboid.width, cuboid.height),
                    Net::new(cuboid.depth, cuboid.height),
                    Net::new(cuboid.width, cuboid.height),
                    Net::new(cuboid.width, cuboid.depth),
                ],
            })
            .collect();

        // The first instruction is to add the first square.
        let queue = vec![Instruction {
            net_pos: Pos::new(middle_x, middle_y),
            face_positions: cuboids.iter().copied().map(FacePos::new).collect(),
            followup_index: None,
        }];

        Ok(Self {
            net,
            surfaces,
            queue,
            index: 0,
            area: 0,
            target_area: cuboids[0].surface_area(),
            prev_nets: Vec::new(),
        })
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
        // println!("{}\n", self.net);
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
            && [Left, Up, Right, Down].into_iter().all(|direction| {
                // Make sure that the new square doesn't make it so that two squares that are
                // next to each other on the net aren't next to each other on the cuboid's
                // surface, since that would require extra cuts to the net which are disallowed.
                // Note that it's perfectly fine (and required, in fact) for to adjacent spots
                // on the surface to not be adjacent on the net.
                if let Some(net_pos) = net_pos.moved_in(direction, self.net.size()) {
                    if self.net.filled(net_pos)
                        && zip(&self.surfaces, &face_positions)
                            .any(|(surface, &pos)| !surface.filled(pos.moved_in(direction)))
                    {
                        return false;
                    }
                }
                true
            })
        {
            // The space is free, so we fill it.
            self.set_square(net_pos, &face_positions, true);
            // If we've just reached a net we've already tried, revert.
            if self.prev_nets.contains(&self.net) {
                self.set_square(net_pos, &face_positions, false);
            } else {
                self.prev_nets.push(self.net.shrink());
                self.queue[self.index].followup_index = Some(next_index);

                // Add the new things we can do from here to the queue.
                self.queue
                    .extend([Left, Up, Right, Down].into_iter().filter_map(|direction| {
                        Some(Instruction {
                            net_pos: net_pos.moved_in(direction, self.net.size())?,
                            face_positions: face_positions
                                .iter()
                                .map(|&pos| pos.moved_in(direction))
                                .collect(),
                            followup_index: None,
                        })
                    }));
            }
        }
        self.index += 1;
    }
}

impl Iterator for NetFinder {
    type Item = Net;

    fn next(&mut self) -> Option<Self::Item> {
        // We start from the bottom-left of the bottom face of the cuboid, and work our
        // way out from there. The net is aligned with the bottom face, so left, right,
        // up, down on the net correspond to the same directions on the bottom face.
        // From there they continue up other faces, e.g. left on the net can become up
        // on the left face.
        // We think of directions on each face as they appear from the inside.

        while self.area < self.target_area {
            while self.index >= self.queue.len() {
                // We've exhausted the queue and haven't found a solution, so we
                // must have done something wrong. Backtrack.
                if !self.backtrack() {
                    return None;
                }
            }

            // Evaluate the next instruction in the queue.
            self.evaluate_instruction();
        }

        let net = self.net.shrink();
        self.backtrack();
        Some(net)
    }
}
