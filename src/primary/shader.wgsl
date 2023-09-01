// This is what we'd ideally write, but unfortunately wgpu doesn't yet support
// pipeline overrides or constant expressions so the CPU inserts them manually:
//
// override num_cuboids: u32;
// override area: u32;
// override trie_capacity: u32;
//
// const net_width = 2u * area + 1u;
// const net_len = net_width * net_width;
//
// const num_squares = area;
// const num_cursors = 4u * num_squares;
// // There can be at most 4 instructions trying to set each square, one coming
// // from each direction.
// const queue_capacity = num_cursors;
//
// /// The index of the start of the page in the trie where you should start
// /// searching.
// const trie_start = num_cursors;
//
// // We need 1 32-bit word per 32 squares, rounded up.
// const surface_words = (num_squares + 31u) / 32u;

// Use a `vec4<Cursor>` rather than just storing 4 cursors directly to satisfy
// the requirement that arrays in uniform buffers have a stride of at least 16
// bytes.
@group(0) @binding(0) var<uniform> neighbour_lookups: array<array<vec4<u32>, num_squares>, num_cuboids>;

/// Returns the neighbour of a cursor in a given direction on the given cuboid.
fn cursor_neighbour(cuboid: u32, cursor: u32, direction: u32) -> u32 {
    let square = cursor >> 2u;
    let old_orientation = cursor & 3u;
    let unrotated = neighbour_lookups[cuboid][square][(old_orientation + direction) & 3u];
    let new_orientation = (old_orientation + (unrotated & 3u)) & 3u;
    return (unrotated & 0xfffffffcu) | new_orientation;
}

/// A single unit of work for trying combinations of net squares.
struct Finder {
    /// A trie encoding a set of mappings to skip, because they're already
    /// covered by other `Finder`s.
    skip: array<u32, trie_capacity>,

    /// The 'queue' of instructions to try and run, except that it's not really
    /// a queue: we don't remove instructions after running them, so that we can
    /// backtrack and try not running them instead.
    queue: Queue,
    /// A list of instructions in the queue marked as 'potential' instructions.
    potential: array<u32, queue_capacity>,
    /// The length of `potential`.
    potential_len: u32,

    /// The index of the instruction that we're going to try and run next.
    index: u32,
    /// The index of the first instruction that isn't fixed: when we attempt to
    /// backtrack past this, the `Finder` is finished.
    base_index: u32,

    /// Bitfields encoding which squares on the surfaces of each cuboid are
    /// filled.
    surfaces: array<array<u32, surface_words>, num_cuboids>,
}

/// A list of instructions.
struct Queue {
    items: array<Instruction, queue_capacity>,
    len: u32,
}

/// An instruction to set a position on the net, and map it to a particular
/// cursor on each cuboid.
struct Instruction {
    /// The position on the net this instruction sets, as an index into
    /// `pos_states`.
    net_pos: u32,
    /// The cursors on the surfaces of each cuboid that this instruction maps
    /// `net_pos` to.
    mapping: array<u32, num_cuboids>,
    /// The index in the queue of the first followup instruction added as a
    /// result of this instruction, or 0 if it hasn't been run.
    /// (No instruction could actually have a followup index of 0, since that'd
    /// mean it had an index less than 0.)
    followup_index: u32,
}

struct PosState {
    // note: all these u32s could actually be u8s, so try bit-packing them later
    // as an optimisation.
    /// The state that this position is in, one of:
    /// - `UNKNOWN`: we don't know anything about this position yet.
    /// - `KNOWN`: if this position is filled, it has to map to `mapping`.
    /// - `INVALID`: there are two neighbouring instructions that expect this
    ///   position to map to two different things, so it can't be filled.
    /// - `FILLED`: this position is filled.
    state: u32,
    /// The earliest mapping that an instruction expected this position to map
    /// to. So, in `KNOWN` or `FILLED` state this is just the one mapping that
    /// this position has to / is mapped to, and in `INVALID` state this is the
    /// first of multiple mappings that this position was expected to map to.
    mapping: array<u32, num_cuboids>,
    /// The index of the first instruction that expected this position to map to
    /// `mapping`.
    /// Note that this is *not* the instruction that actually wants to map this
    /// position to said mapping; this is its parent, since we want this
    /// `PosState` to get set even if the instruction's invalid.
    setter: u32,
    /// If `state` is `INVALID`, the index of the first instruction to expect
    /// this position to map to something other than `mapping`.
    conflict: u32,
}

const UNKNOWN = 0u;
const KNOWN = 1u;
const INVALID = 2u;
const FILLED = 3u;

@group(0) @binding(1) var<storage, read_write> finders: array<Finder>;

/// Returns whether or not an instruction should be skipped.
fn skip(trie: ptr<function, array<u32, trie_capacity>>, instruction: Instruction) -> bool {
    var mapping = instruction.mapping;

    var page = trie_start;
    for (var i = 0u; i < num_cuboids - 1u; i++) {
        let index = page + mapping[i];
        page = (*trie)[index];
    }
    let bit_index = mapping[num_cuboids - 1u];
    let word = (*trie)[bit_index / 32u];
    return bool(word & (1u << (bit_index % 32u)));
}

/// Returns the neighbour of an instruction in a given direction.
fn instruction_neighbour(instruction: Instruction, direction: u32) -> Instruction {
    var result = instruction;

    // If `direction` is left/right (even), we need to offset the index by 1,
    // otherwise we need to offset it by `net_width`.
    let magnitude = select(net_width, 1u, direction % 2u == 0u);
    // Then, if `direction` is left or down, we need to subtract that offset,
    // otherwise we add it. Left and down are represented as 0 and 3, so
    // subtracting 1 means they become 2^32 - 1 and 2 respectively, and so we
    // can just check if that's >= 2.
    result.net_pos = select(result.net_pos + magnitude, result.net_pos - magnitude, direction - 1u >= 2u);

    for (var i = 0u; i < num_cuboids; i++) {
        result.mapping[i] = cursor_neighbour(i, result.mapping[i], direction);
    }

    // New instructions should always be marked as unrun.
    result.followup_index = 0u;

    return result;
}

/// Returns whether the given square is filled on the given surface.
fn filled(surface: ptr<function, array<u32, surface_words>>, square: u32) -> bool {
    return bool((*surface)[square / 32u] & (1u << (square % 32u)));
}

/// Sets whether the given square is filled on the given surface.
fn set_filled(surface: ptr<function, array<u32, surface_words>>, square: u32, value: bool) {
    (*surface)[square / 32u] &= ~(1u << (square % 32u));
    (*surface)[square / 32u] |= (u32(value) << (square % 32u));
}

/// A potential solution net.
struct MaybeSolution {
    /// The list of instructions that must definitely be run to produce this
    /// solution.
    completed: Queue,
    /// The list of instructions that might need to potentially be run to
    /// produce this solution.
    potential: Queue,
}

/// A list of `MaybeSolution`s.
struct MaybeSolutions {
    /// The length of `solutions`.
    len: atomic<u32>,
    /// The list of solutions.
    items: array<MaybeSolution>,
}

@group(0) @binding(2) var<storage, read_write> pos_states: array<array<PosState, net_len>>;

@group(0) @binding(3) var<storage, read_write> maybe_solutions: MaybeSolutions;

@compute @workgroup_size(64)
fn run_finder(@builtin(global_invocation_id) id: vec3<u32>) {
    var finder = finders[id.x];
    let pos_states_idx = id.x;

    let pos_states = &pos_states[pos_states_idx];
    // Fill in `pos_states`.
    // First off, the first instruction in the queue gets special treatement.
    let first_instruction = finder.queue.items[0];
    (*pos_states)[first_instruction.net_pos].state = KNOWN;
    (*pos_states)[first_instruction.net_pos].mapping = first_instruction.mapping;
    // This is incorrect, but works since this only ever does anything if this
    // is set to the index of a neighbouring instruction, and the first
    // instruction isn't a neighbour of itself.
    (*pos_states)[first_instruction.net_pos].setter = 0u;
    // Then for everything else, we can just re-run all the instructions in the
    // queue that have already been run.
    //
    // (This also needlessly re-fills in `finder.surfaces` but it doesn't really
    // matter.)
    for (var i = 0u; i < finder.queue.len; i++) {
        if finder.queue.items[i].followup_index != 0u {
            run_instruction(&finder, pos_states_idx, i);
        }
    }

    for (var i = 0u; i < 10000u; i++) {
        if finder.index < finder.queue.len {
            handle_instruction(&finder, pos_states_idx);
        } else {
            // We've reached the end of the queue, so we might have a solution!
            // However, the very very common case is that we definitely don't
            // because there are squares that no potential instructions cover.
            if covers_surface(&finder, pos_states_idx) {
                if !write_solution(&finder, pos_states_idx) {
                    // Stop searching if there's no room for more solutions.
                    //
                    // Don't backtrack, since we want to still be at the end of
                    // the queue next time and try to write this solution again.
                    break;
                }
            }

            if !backtrack(&finder, pos_states_idx) {
                break;
            }
        }
    }

    finders[id.x] = finder;
}

/// Attempts to run the next instruction in `finder`'s queue.
fn handle_instruction(finder: ptr<function, Finder>, pos_states_idx: u32) {
    let instruction = &(*finder).queue.items[(*finder).index];
    if valid(finder, pos_states_idx, *instruction) {
        let old_len = (*finder).queue.len;
        // Add all this instruction's follow-up instructions to the queue.
        for (var direction = 0u; direction < 4u; direction++) {
            let instruction = instruction_neighbour(*instruction, direction);
            if pos_states[pos_states_idx][instruction.net_pos].state == UNKNOWN && valid(finder, pos_states_idx, instruction) && !skip(&(*finder).skip, instruction) {
                (*finder).queue.items[(*finder).queue.len] = instruction;
                (*finder).queue.len += 1u;
            }
        }

        // Only if it had follow-up instructions do we actually run it.
        if (*finder).queue.len > old_len {
            (*instruction).followup_index = old_len;
            run_instruction(finder, pos_states_idx, (*finder).index);
        } else {
            // Otherwise mark it as potential.
            (*finder).potential[(*finder).potential_len] = (*finder).index;
            (*finder).potential_len += 1u;
        }
    }
    (*finder).index += 1u;
}

/// Runs the instruction at index `index` in `finder`'s queue.
fn run_instruction(finder: ptr<function, Finder>, pos_states_idx: u32, index: u32) {
    var instruction = (*finder).queue.items[index];

    // Fill in the squares that the instruction sets.
    pos_states[pos_states_idx][instruction.net_pos].state = FILLED;
    for (var i = 0u; i < num_cuboids; i++) {
        // Functions aren't allowed to take pointers in the storage
        // address space, so we have to make a copy.
        set_filled(&(*finder).surfaces[i], instruction.mapping[i] >> 2u, true);
    }

    // Update the `pos_states` of neighbouring squares.
    for (var direction = 0u; direction < 4u; direction++) {
        let instruction = instruction_neighbour(instruction, direction);
        let pos_state = &pos_states[pos_states_idx][instruction.net_pos];
        if (*pos_state).state == UNKNOWN {
            (*pos_state).mapping = instruction.mapping;
            (*pos_state).state = KNOWN;
            (*pos_state).setter = index;
        } else if (*pos_state).state == KNOWN && !mapping_eq(instruction.mapping, (*pos_state).mapping) {
            (*pos_state).state = INVALID;
            (*pos_state).conflict = index;
        }
    }
}

/// Attempts to undo the last instruction run by `finder`.
///
/// Returns true on success, and false otherwise, which happens if `finder` has
/// no run instructions left; in that case, it's done!
fn backtrack(finder: ptr<function, Finder>, pos_states_idx: u32) -> bool {
    // Find the instruction that was last run.
    var last_run = (*finder).queue.len - 1u;
    if last_run >= queue_capacity {
        // this should never happen but i've been burned too many times
        return false;
    }
    while (*finder).queue.items[last_run].followup_index == 0u {
        if last_run <= (*finder).base_index {
            // There are no completed instructions left past `base_index`;
            // we're done!
            return false;
        }
        last_run -= 1u;
    }

    let instruction = &(*finder).queue.items[last_run];
    // Remove all `instruction`'s follow-up instructions from the queue and mark
    // it as not run.
    (*finder).queue.len = (*instruction).followup_index;
    (*instruction).followup_index = 0u;

    // Unmark its squares on the net and the surface as filled.
    pos_states[pos_states_idx][(*instruction).net_pos].state = KNOWN;
    for (var i = 0u; i < num_cuboids; i++) {
        var surface = (*finder).surfaces[i];
        set_filled(&surface, (*instruction).mapping[i] >> 2u, false);
        (*finder).surfaces[i] = surface;
    }

    // Update the `pos_states` of all the neighbouring net positions.
    for (var direction = 0u; direction < 4u; direction++) {
        let instruction = instruction_neighbour(*instruction, direction);
        let pos_state = &pos_states[pos_states_idx][instruction.net_pos];
        // If the instruction we just reverted was the one that advanced this
        // `PosState` to the `KNOWN` or `INVALID` state, move it back to the
        // previous state.
        if ((*pos_state).state == INVALID && (*pos_state).conflict == last_run) || ((*pos_state).state == KNOWN && (*pos_state).setter == last_run) {
            (*pos_state).state -= 1u;
        }
    }

    // Also un-mark any instructions that come after this instruction as
    // potential, since they would otherwise have been backtracked first.
    while (*finder).potential_len > 0u && (*finder).potential[(*finder).potential_len - 1u] >= last_run {
        (*finder).potential_len -= 1u;
    }

    // Now we continue executing from after the instruction we backtracked.
    (*finder).index = last_run + 1u;

    return true;
}

/// Returns whether an instruction is valid. Note that this does *not* consider whether it's skipped.
fn valid(finder: ptr<function, Finder>, pos_states_idx: u32, instruction: Instruction) -> bool {
    // wgpu doesn't support and-assign for bools yet so use an integer.
    var result = 1u;

    result &= u32(pos_states[pos_states_idx][instruction.net_pos].state != INVALID);

    var mapping = instruction.mapping;
    for (var i = 0u; i < num_cuboids; i++) {
        var surface = (*finder).surfaces[i];
        result &= u32(!filled(&surface, mapping[i] >> 2u));
    }

    return bool(result);
}

/// Returns whether two mappings are equal.
fn mapping_eq(a: array<u32, num_cuboids>, b: array<u32, num_cuboids>) -> bool {
    var a2 = a;
    var b2 = b;
    var result = 1u;
    for (var i = 0u; i < num_cuboids; i++) {
        result &= u32(a2[i] == b2[i]);
    }
    return bool(result);
}

fn covers_surface(finder: ptr<function, Finder>, pos_states_idx: u32) -> bool {
    var surfaces = (*finder).surfaces;

    // Return early in the common case that there aren't enough potential
    // instructions to possibly fill the remaining squares.
    if num_filled(&surfaces[0]) + (*finder).potential_len < area {
        return false;
    }

    for (var i = 0u; i < (*finder).potential_len; i++) {
        let index = (*finder).potential[i];
        var instruction = (*finder).queue.items[index];
        if valid(finder, pos_states_idx, instruction) {
            for (var i = 0u; i < num_cuboids; i++) {
                set_filled(&surfaces[i], instruction.mapping[i] >> 2u, true);
            }
        }
    }

    for (var i = 0u; i < num_cuboids; i++) {
        if num_filled(&surfaces[i]) < area {
            return false;
        }
    }

    return true;
}

/// Returns the number of filled squares on a surface.
fn num_filled(surface: ptr<function, array<u32, surface_words>>) -> u32 {
    var total = 0u;
    for (var i = 0u; i < surface_words; i++) {
        total += countOneBits((*surface)[i]);
    }
    return total;
}

/// Writes the current state of `finder` as a solution to `maybe_solutions`.
///
/// Returns whether writing was successful; if false is returned, it means that
/// there was no more room.
fn write_solution(finder: ptr<function, Finder>, pos_states_idx: u32) -> bool {
    let index = atomicAdd(&maybe_solutions.len, 1u);
    if index >= arrayLength(&maybe_solutions.items) {
        // Oops, there's not actually that much room. Undo.
        atomicStore(&maybe_solutions.len, arrayLength(&maybe_solutions.items));
        return false;
    }
    let sol = &maybe_solutions.items[index];
    (*sol).completed.len = 0u;
    (*sol).potential.len = 0u;

    for (var i = 0u; i < (*finder).queue.len; i++) {
        let instruction = (*finder).queue.items[i];
        if instruction.followup_index != 0u {
            (*sol).completed.items[(*sol).completed.len] = instruction;
            (*sol).completed.len += 1u;
        }
    }

    for (var i = 0u; i < (*finder).potential_len; i++) {
        let index = (*finder).potential[i];
        let instruction = (*finder).queue.items[index];
        if valid(finder, pos_states_idx, instruction) {
            (*sol).potential.items[(*sol).potential.len] = instruction;
            (*sol).potential.len += 1u;
        }
    }

    return true;
}
