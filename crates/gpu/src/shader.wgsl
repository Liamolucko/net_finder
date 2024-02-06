// This is what we'd ideally write, but unfortunately wgpu doesn't yet support
// pipeline overrides or constant expressions so the CPU inserts them manually:
//
// override num_cuboids: u32;
// override area: u32;
// override trie_capacity: u32;
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
    /// The set of instructions in `queue`, using a weird custom encoding.
    ///
    /// The encoding is similar to `InstructionSet` but slightly different.
    /// Firstly, there's one `vec4<u32>` corresponding to every square on the
    /// first cuboid. Each `u32` contains the net position of one instruction
    /// that sets that square.
    ///
    /// However, a net position only takes up 16 bits. Since we have no choice
    /// but to use full `u32`s, we use the upper bits to do two things at once:
    /// - The MSB of any position represents whether that spot is filled.
    /// - The next 3 bits of `positions.x` store the number of filled spots.
    queued: array<vec4<u32>, num_squares>,
}

/// A list of instructions.
struct Queue {
    items: array<Instruction, queue_capacity>,
    len: u32,
}

/// An instruction to set a position on the net, and map it to a particular
/// cursor on each cuboid.
struct Instruction {
    /// The position on the net this instruction sets. Encoded as
    ///     00000000_00000000_xxxxxxxx_yyyyyyyy.
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

    // This is a little trick I noticed rustc using to optimise small match
    // statements.
    let x_offset = (0x000100ffu >> (8u * direction)) & 0xffu;
    let y_offset = (0xff000100u >> (8u * direction)) & 0xffu;

    let x = ((result.net_pos >> 8u) + x_offset) & 0xffu;
    let y = (result.net_pos + y_offset) & 0xffu;
    result.net_pos = (x << 8u) | y;

    for (var i = 0u; i < num_cuboids; i++) {
        result.mapping[i] = cursor_neighbour(i, result.mapping[i], direction);
    }

    // New instructions should always be marked as unrun.
    result.followup_index = 0u;

    return result;
}

/// Inserts an instruction into `Finder::queued` and returns whether it was not
/// already contained within the set.
fn insert(queued: ptr<function, array<vec4<u32>, num_squares>>, instruction: Instruction) -> bool {
    let positions = &(*queued)[instruction.mapping[0] >> 2u];
    let tagged_pos = instruction.net_pos | 0x80000000u;
    let exists = any((*positions & vec4(0x8fffffffu)) == vec4(tagged_pos));
    if !exists {
        let length = ((*positions).x >> 28u) & 0x7u;
        (*positions)[length] = tagged_pos;
        // Increment the length.
        (*positions).x += 0x10000000u;
    }
    return !exists;
}

/// Removes the last added instruction which sets the same square on the first
/// cuboid as `instruction` from `Finder::queued`.
///
/// This would be rather a weird thing to do in general, but we always remove
/// instructions from the queue in the opposite order we add them (it's a stack!)
/// so it's fine.
fn remove(queued: ptr<function, array<vec4<u32>, num_squares>>, instruction: Instruction) {
    let positions = &(*queued)[instruction.mapping[0] >> 2u];
    // Decrement the length.
    (*positions).x -= 0x10000000u;
    // Set the entry we're removing back to 0.
    let length = ((*positions).x >> 28u) & 0x7u;
    (*positions)[length] = 0u;
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

@group(0) @binding(2) var<storage, read_write> maybe_solutions: MaybeSolutions;

@compute @workgroup_size(64)
fn run_finder(@builtin(global_invocation_id) id: vec3<u32>) {
    var finder = finders[id.x];

    for (var i = 0u; i < 10000u; i++) {
        if finder.index < finder.queue.len {
            handle_instruction(&finder);
        } else {
            // We've reached the end of the queue, so we might have a solution!
            // However, the very very common case is that we definitely don't
            // because there are squares that no potential instructions cover.
            if covers_surface(&finder) {
                if !write_solution(&finder) {
                    // Stop searching if there's no room for more solutions.
                    //
                    // Don't backtrack, since we want to still be at the end of
                    // the queue next time and try to write this solution again.
                    break;
                }
            }

            if !backtrack(&finder) {
                break;
            }
        }
    }

    finders[id.x] = finder;
}

/// Attempts to run the next instruction in `finder`'s queue.
fn handle_instruction(finder: ptr<function, Finder>) {
    let instruction = &(*finder).queue.items[(*finder).index];
    if valid(finder, *instruction) {
        let old_len = (*finder).queue.len;
        // Add all this instruction's follow-up instructions to the queue.
        for (var direction = 0u; direction < 4u; direction++) {
            let instruction = instruction_neighbour(*instruction, direction);
            if valid(finder, instruction) && !skip(&(*finder).skip, instruction) {
                if insert(&(*finder).queued, instruction) {
                    (*finder).queue.items[(*finder).queue.len] = instruction;
                    (*finder).queue.len += 1u;
                }
            }
        }

        // Only if it had follow-up instructions do we actually run it.
        if (*finder).queue.len > old_len {
            (*instruction).followup_index = old_len;

            // Fill in the squares that the instruction sets.
            for (var i = 0u; i < num_cuboids; i++) {
                // Functions aren't allowed to take pointers in the storage
                // address space, so we have to make a copy.
                set_filled(&(*finder).surfaces[i], (*instruction).mapping[i] >> 2u, true);
            }
        } else {
            // Otherwise mark it as potential.
            (*finder).potential[(*finder).potential_len] = (*finder).index;
            (*finder).potential_len += 1u;
        }
    }
    (*finder).index += 1u;
}

/// Attempts to undo the last instruction run by `finder`.
///
/// Returns true on success, and false otherwise, which happens if `finder` has
/// no run instructions left; in that case, it's done!
fn backtrack(finder: ptr<function, Finder>) -> bool {
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
    for (var i = (*finder).queue.len - 1u; i >= (*instruction).followup_index; i--) {
        remove(&(*finder).queued, (*finder).queue.items[i]);
    }
    (*finder).queue.len = (*instruction).followup_index;
    (*instruction).followup_index = 0u;

    // Unmark its squares on the surface as filled.
    for (var i = 0u; i < num_cuboids; i++) {
        set_filled(&(*finder).surfaces[i], (*instruction).mapping[i] >> 2u, false);
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
fn valid(finder: ptr<function, Finder>, instruction: Instruction) -> bool {
    // wgpu doesn't support and-assign for bools yet so use an integer.
    var result = 1u;

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

fn covers_surface(finder: ptr<function, Finder>) -> bool {
    var surfaces = (*finder).surfaces;

    // Return early in the common case that there aren't enough potential
    // instructions to possibly fill the remaining squares.
    if num_filled(&surfaces[0]) + (*finder).potential_len < area {
        return false;
    }

    for (var i = 0u; i < (*finder).potential_len; i++) {
        let index = (*finder).potential[i];
        var instruction = (*finder).queue.items[index];
        if valid(finder, instruction) {
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
fn write_solution(finder: ptr<function, Finder>) -> bool {
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
        if valid(finder, instruction) {
            (*sol).potential.items[(*sol).potential.len] = instruction;
            (*sol).potential.len += 1u;
        }
    }

    return true;
}
