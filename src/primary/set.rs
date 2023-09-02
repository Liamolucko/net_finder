use serde::{Deserialize, Serialize};

use crate::Pos;

use super::Instruction;

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
    elements: Vec<(u8, [Pos; 4])>,
}

impl<const CUBOIDS: usize> InstructionSet<CUBOIDS> {
    /// Creates a new `InstructionSet` which can store instructions operating on
    /// cuboids of the given surface area.
    pub(super) fn new(area: usize) -> Self {
        Self {
            elements: vec![(0, [Pos::new(0, 0); 4]); area],
        }
    }

    /// Inserts an instruction into the set and returns whether it was already
    /// contained within the set.
    pub(super) fn insert(&mut self, instruction: Instruction<CUBOIDS>) -> bool {
        let (count, positions) =
            &mut self.elements[usize::from(instruction.mapping.cursors[0].square().0)];
        let new = !positions[..usize::from(*count)]
            .iter()
            .any(|&pos| pos == instruction.net_pos);
        if new {
            positions[usize::from(*count)] = instruction.net_pos;
            *count += 1;
        }
        new
    }

    /// Removes an instruction from the set.
    ///
    /// Does nothing if the instruction isn't in the set.
    pub(super) fn remove(&mut self, instruction: &Instruction<CUBOIDS>) {
        let (count, positions) =
            &mut self.elements[usize::from(instruction.mapping.cursors[0].square().0)];
        if let Some(index) = positions[..usize::from(*count)]
            .iter()
            .position(|&pos| pos == instruction.net_pos)
        {
            *count -= 1;
            for i in index..usize::from(*count) {
                positions[i] = positions[i + 1];
            }
        }
    }
}
