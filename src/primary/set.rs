use serde::{Deserialize, Serialize};

use super::Instruction;

/// Builds a `u64` consisting of 4 copies of `value`.
#[inline]
fn repeat(value: u16) -> u64 {
    0x0001_0001_0001_0001 * (value as u64)
}

/// Returns the index of the first 16-bit chunk of `word` equal to `value`
/// (from LSB to MSB), or 4 if there is none.
///
/// Based on https://graphics.stanford.edu/~seander/bithacks.html#ValueInWord,
/// which I found via. hashbrown.
#[inline]
fn find_u16(word: u64, value: u16) -> u32 {
    let cmp = word ^ repeat(value);
    (cmp.wrapping_sub(repeat(0x0001)) & !cmp & repeat(0x8000)).trailing_zeros() / 16
}

/// A set of queued instructions.
///
/// Well, not really: you can't actually get the instructions back out of it.
/// That's not needed anyway, since this should only ever be a faster lookup for
/// whether or not an instruction's in the queue.
///
/// It also has some other limitations which work for the case of checking what
/// instructions are in the queue:
/// - There can only be at most 4 instructions that set one square, since each
///   instruction must come from another instruction and in the worst-case
///   scenario, all 4 squares around a square are set.
/// - There can only be one instruction that sets a given position on the net.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct InstructionSet<const CUBOIDS: usize> {
    /// For each square on the first cuboid, the net positions of the up-to-4
    /// instructions that set that square. Represented as first the number of
    /// them, and then the actual positions.
    elements: Vec<(u32, u64)>,
}

impl<const CUBOIDS: usize> InstructionSet<CUBOIDS> {
    /// Creates a new `InstructionSet` which can store instructions operating on
    /// cuboids of the given surface area.
    pub(super) fn new(area: usize) -> Self {
        Self {
            elements: vec![(0, 0); area],
        }
    }

    /// Inserts an instruction into the set and returns whether it was already
    /// contained within the set.
    #[inline]
    pub(super) fn insert(&mut self, instruction: Instruction<CUBOIDS>) -> bool {
        let (count, word) =
            &mut self.elements[usize::from(instruction.mapping.cursors[0].square().0)];
        let encoded_pos: u16 = bytemuck::cast(instruction.net_pos);
        let new = find_u16(*word, encoded_pos) >= *count;
        if new {
            *word |= (encoded_pos as u64) << 16 * *count;
            *count += 1;
        }
        new
    }

    /// Removes an instruction from the set.
    ///
    /// Does nothing if the instruction isn't in the set.
    #[inline]
    pub(super) fn remove(&mut self, instruction: &Instruction<CUBOIDS>) {
        let (count, word) =
            &mut self.elements[usize::from(instruction.mapping.cursors[0].square().0)];
        let encoded_pos: u16 = bytemuck::cast(instruction.net_pos);
        let index = find_u16(*word, encoded_pos);
        if index < *count {
            let after_mask = u64::MAX << 16 * index;
            let before_mask = !after_mask;
            *word = (*word & before_mask) | ((*word >> 16) & after_mask);
            *count -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn find_u16() {
        assert_eq!(super::find_u16(0, 1), 4);
        assert_eq!(super::find_u16(0, 0), 0);
        assert_eq!(super::find_u16(0x00000000_0000beef, 0xbeef), 0);
        assert_eq!(super::find_u16(0x00000000_beef0000, 0xbeef), 1);
        assert_eq!(super::find_u16(0x0000beef_00000000, 0xbeef), 2);
        assert_eq!(super::find_u16(0xbeef0000_00000000, 0xbeef), 3);
    }
}
