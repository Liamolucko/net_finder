// These are replaced by the CPU: I wanted to use pipeline overrides for this,
// but it turns out you can't use pipeline overrides for array lengths within
// structs.
const num_cuboids: u32 = 2;
const area: u32 = 22;

const num_squares = area;
const num_cursors = 4 * num_squares;
// There can be at most 4 instructions trying to set each square, one coming
// from each direction.
const queue_capacity = num_cursors;

const net_size = area;
const net_words = (net_size * net_size + 31) / 32;

// We need 1 32-bit word per 32 squares, rounded up.
const surface_words = (num_squares + 31) / 32;

const decision_words = (queue_capacity + 31) / 32;

// Use a `vec4<Cursor>` rather than just storing 4 cursors directly to satisfy
// the requirement that arrays in uniform buffers have a stride of at least 16
// bytes.
@group(0) @binding(0) var<uniform> neighbour_lookups: array<array<vec4<u32>, num_squares>, num_cuboids>;

/// Returns the neighbour of a cursor in a given direction on the given cuboid.
fn cursor_neighbour(cuboid: u32, cursor: u32, direction: u32) -> u32 {
    let square = cursor >> 2u;
    let old_orientation = cursor & 3;
    let unrotated = neighbour_lookups[cuboid][square][(old_orientation + direction) & 3];
    let new_orientation = (old_orientation + (unrotated & 3)) & 3;
    return (unrotated & 0xfffffffc) | new_orientation;
}

/// An instruction to set a position on the net, and map it to a particular
/// cursor on each cuboid.
struct Instruction {
    /// The position on the net this instruction sets. Encoded as
    ///     00000000_00000000_0000xxxx_xxyyyyyy.
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

/// Returns the neighbour of an instruction in a given direction.
fn instruction_neighbour(instruction: Instruction, direction: u32) -> Instruction {
    var result = instruction;

    let old_x = result.net_pos >> 6u;
    let old_y = result.net_pos & 0x3f;

    // This is a little trick I noticed rustc using to optimise small match
    // statements.
    let x_offset = (0x0001003fu >> (8 * direction)) & 0xff;
    let y_offset = (0x3f000100u >> (8 * direction)) & 0xff;

    let x_edge = direction % 2 == 0 && old_x == select(0u, area - 1, direction == 2);
    let y_edge = direction % 2 == 1 && old_y == select(area - 1, 0u, direction == 3);

    let x = select((old_x + x_offset) & 0x3f, select(area - 1, 0u, direction == 2), x_edge);
    let y = select((old_y + y_offset) & 0x3f, select(0u, area - 1, direction == 3), y_edge);
    result.net_pos = (x << 6u) | y;

    for (var i = 0u; i < num_cuboids; i++) {
        result.mapping[i] = cursor_neighbour(i, result.mapping[i], direction);
    }

    // New instructions should always be marked as unrun.
    result.followup_index = 0;

    return result;
}

// A mapping from cursors to their classes.
//
// Because of the stride restrictions, the result is looked up by first getting
// the right `vec4` using the cursor's square, then indexing into that `vec4`
// with its orientation.
@group(0) @binding(1) var<uniform> class_lookups: array<array<vec4<u32>, num_squares>, num_cuboids>;

// Returns the class of a cursor.
fn cursor_class(cuboid: u32, cursor: u32) -> u32 {
    return class_lookups[cuboid][cursor >> 2u][cursor & 3];
}

// Each `vec4` contains the class you get by undoing each of the 8 possible
// transforms a class in the fixed family might have.
//
// There are actually two entries for each transform, though: sometimes, there
// are multiple transformations which have the same effect on the fixed family.
// However, these transformations don't necessarily have the same effect on the
// other classes in a mapping, meaning that we can end up with multiple
// equivalent mappings which use the fixed class.
//
// Since the goal is to place where in the ordering of equivalence classes this
// mapping's equivalence class falls, and we sort them by the smallest mapping
// index in that class which contains the fixed class, this means we need to
// choose the one with the smallest mapping index out of that set.
//
// For all the scenarios we care about, each transformation on the fixed family
// only has one other transformation that's equivalent to it, so we store what
// you get by undoing either of those two transformations.
//
// Each u32 is split into a u16 for each transformation to be undone, and each
// u16 is split into two u8s for the two different transforms to undo.
//
// TODO: are we actually gaining anything by putting in cursors here rather than
// classes?
@group(0) @binding(2) var<uniform> undo_lookups: array<array<vec4<u32>, num_cursors>, num_cuboids - 1>;

/// Returns whether or not an instruction should be skipped.
fn skip(fixed_class: u32, start_mapping_index: u32, instruction: Instruction) -> bool {
    var mapping = instruction.mapping;
    var mapping_indices = array<u32, 2>(0, 0);
    let class0 = cursor_class(0, mapping[0]);
    // TODO: this also needs to be fixed to use the correct transform mask.
    let to_undo = class0 & 7;
    for (var cuboid = 1u; cuboid < num_cuboids; cuboid++) {
        let choices = (undo_lookups[cuboid - 1][mapping[cuboid]][to_undo >> 1u] >> (16 * (to_undo & 1))) & 0xffff;
        for (var choice = 0u; choice < 2; choice++) {
            mapping_indices[choice] <<= 6;
            mapping_indices[choice] |= (choices >> (8 * choice)) & 0xff;
        }
    }

    // TODO: this assumes we always have 3 transform bits, which is wrong.
    // we can fix this by adding a `fixed_family_mask` field to `FinderInfo` and `Finder`.
    return (class0 & 0x38) == fixed_class && (mapping_indices[0] < start_mapping_index || mapping_indices[1] < start_mapping_index);
}

/// A single unit of work for trying combinations of net squares.
struct Finder {
    fixed_class: u32,
    start_mapping_index: u32,

    /// The 'queue' of instructions to try and run, except that it's not really
    /// a queue: we don't remove instructions after running them, so that we can
    /// backtrack and try not running them instead.
    // TODO: try moving this to a var<private> to reduce register spills
    queue: array<Instruction, queue_capacity>,
    /// The length of `queue`.
    queue_len: u32,
    /// A list of instructions in the queue marked as 'potential' instructions.
    potential: array<u32, queue_capacity>,
    /// The length of `potential`.
    potential_len: u32,

    /// The index of the instruction that we're going to try and run next.
    index: u32,
    /// The index of the first instruction that isn't fixed: when we attempt to
    /// backtrack past this, the `Finder` is finished.
    base_index: u32,

    /// A bitfield of which net squares have instructions in the queue that are
    /// trying to set them.
    net: array<u32, net_words>,
    /// Bitfields encoding which squares on the surfaces of each cuboid are
    /// filled.
    surfaces: array<array<u32, surface_words>, num_cuboids>,
}

/// Returns whether an instruction is in the queue.
fn queued(finder: ptr<function, Finder>, instruction: Instruction) -> bool {
    let x = instruction.net_pos >> 6u;
    let y = instruction.net_pos & 0x3f;
    let index = area * x + y;
    return bool((*finder).net[index / 32] & (1u << (index % 32)));
}

/// Sets whether an instruction is in the queue.
fn set_queued(finder: ptr<function, Finder>, instruction: Instruction, value: bool) {
    let x = instruction.net_pos >> 6u;
    let y = instruction.net_pos & 0x3f;
    let index = area * x + y;
    (*finder).net[index / 32] &= ~(1u << (index % 32));
    (*finder).net[index / 32] |= (u32(value) << (index % 32));
}

/// Returns whether the given square is filled on the given surface.
fn filled(finder: ptr<function, Finder>, cuboid: u32, square: u32) -> bool {
    return bool((*finder).surfaces[cuboid][square / 32] & (1u << (square % 32)));
}

/// Sets whether the given square is filled on the given surface.
fn set_filled(finder: ptr<function, Finder>, cuboid: u32, square: u32, value: bool) {
    (*finder).surfaces[cuboid][square / 32] &= ~(1u << (square % 32));
    (*finder).surfaces[cuboid][square / 32] |= (u32(value) << (square % 32));
}

struct FinderInfo {
    start_mapping: array<u32, num_cuboids>,
    decisions: array<u32, decision_words>,
    decisions_len: u32,
    base_decision: u32,
}

@group(0) @binding(3) var<storage, read_write> finders: array<FinderInfo>;

/// A list of possible solutions, represented as `FinderInfo`s.
struct MaybeSolutions {
    /// The length of `solutions`.
    len: atomic<u32>,
    /// The list of solutions.
    items: array<FinderInfo>,
}

@group(0) @binding(4) var<storage, read_write> maybe_solutions: MaybeSolutions;

@compute @workgroup_size(64)
fn run_finder(@builtin(global_invocation_id) id: vec3<u32>) {
    let info = finders[id.x];

    let fixed_class = cursor_class(0, info.start_mapping[0]);

    var start_mapping_index = 0u;
    for (var i = 1u; i < num_cuboids; i++) {
        start_mapping_index <<= 6;
        start_mapping_index |= cursor_class(i, info.start_mapping[i]);
    }

    var finder = Finder(
        fixed_class,
        start_mapping_index,
        array<Instruction, queue_capacity>(),
        0,
        array<u32, queue_capacity>(),
        0,
        0,
        0,
        array<u32, net_words>(),
        array<array<u32, surface_words>, num_cuboids>(),
    );

    finder.queue[0] = Instruction(0, info.start_mapping, 0);
    finder.queue_len = 1;
    set_queued(&finder, finder.queue[0], true);

    var decision_index = 0u;
    var base_index_set = false;
    while decision_index < info.decisions_len {
        let old_len = finder.queue_len;
        handle_instruction(&finder);
        if finder.queue_len > old_len {
            // The instruction must have been valid, so this is a decision.
            if decision_index == info.base_decision {
                finder.base_index = finder.index - 1;
                base_index_set = true;
            }

            // If the decision was 0, we need to backtrack.
            if (info.decisions[decision_index / 32] & (1u << decision_index % 32)) == 0 {
                backtrack(&finder);
            }

            decision_index += 1;
        }
    }

    if !base_index_set {
        finder.base_index = finder.index;
    }

    for (var i = 0u; i < 1000u; i++) {
        if finder.index < finder.queue_len {
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

    finders[id.x] = finder_to_info(&finder);
}

/// Attempts to run the next instruction in `finder`'s queue.
fn handle_instruction(finder: ptr<function, Finder>) {
    let instruction = &(*finder).queue[(*finder).index];
    let old_len = (*finder).queue_len;
    if valid(finder, *instruction) && !skip((*finder).fixed_class,(*finder).start_mapping_index, *instruction) {
        // Add all this instruction's follow-up instructions to the queue.
        for (var direction = 0u; direction < 4; direction++) {
            let instruction = instruction_neighbour(*instruction, direction);
            if valid(finder, instruction) && !queued(finder, instruction) {
                (*finder).queue[(*finder).queue_len] = instruction;
                (*finder).queue_len += 1;
                set_queued(finder, instruction, true);
            }
        }

        // Only if it had follow-up instructions do we actually run it.
        if (*finder).queue_len > old_len {
            (*instruction).followup_index = old_len;

            // Fill in the squares that the instruction sets.
            for (var i = 0u; i < num_cuboids; i++) {
                // Functions aren't allowed to take pointers in the storage
                // address space, so we have to make a copy.
                set_filled(finder, i,(*instruction).mapping[i] >> 2u, true);
            }
        } else {
            // Otherwise mark it as potential.
            (*finder).potential[(*finder).potential_len] = (*finder).index;
            (*finder).potential_len += 1;
        }
    }

    if (*finder).queue_len == old_len && (*finder).index == (*finder).base_index {
        // `base_index` is the index of the first instruction we're allowed to
        // backtrack; but we can't backtrack something that wasn't run in the first
        // place, so that's no longer true of this index. Push it forwards onto the next
        // one.
        (*finder).base_index += 1;
    }

    (*finder).index += 1;
}

/// Attempts to undo the last instruction run by `finder`.
///
/// Returns true on success, and false otherwise, which happens if `finder` has
/// no run instructions left; in that case, it's done!
fn backtrack(finder: ptr<function, Finder>) -> bool {
    // Find the instruction that was last run.
    var last_run = (*finder).queue_len - 1;
    if last_run >= queue_capacity {
        // this should never happen but i've been burned too many times
        return false;
    }
    while (*finder).queue[last_run].followup_index == 0 {
        if last_run <= (*finder).base_index {
            // There are no completed instructions left past `base_index`;
            // we're done!
            return false;
        }
        last_run -= 1u;
    }

    let instruction = &(*finder).queue[last_run];
    // Remove all `instruction`'s follow-up instructions from the queue and mark
    // it as not run.
    for (var i = (*finder).queue_len - 1; i >= (*instruction).followup_index; i--) {
        set_queued(finder,(*finder).queue[i], false);
    }
    (*finder).queue_len = (*instruction).followup_index;
    (*instruction).followup_index = 0;
    if (*finder).base_index == last_run {
        // This instruction can't be backtracked again, so it doesn't get to be the base
        // index anymore.
        (*finder).base_index += 1;
    }

    // Unmark its squares on the surface as filled.
    for (var i = 0u; i < num_cuboids; i++) {
        set_filled(finder, i,(*instruction).mapping[i] >> 2u, false);
    }

    // Also un-mark any instructions that come after this instruction as
    // potential, since they would otherwise have been backtracked first.
    while (*finder).potential_len > 0 && (*finder).potential[(*finder).potential_len - 1] >= last_run {
        (*finder).potential_len -= 1;
    }

    // Now we continue executing from after the instruction we backtracked.
    (*finder).index = last_run + 1;

    return true;
}

/// Returns whether an instruction is valid. Note that this does *not* consider whether it's skipped.
fn valid(finder: ptr<function, Finder>, instruction: Instruction) -> bool {
    var result = true;

    var mapping = instruction.mapping;
    for (var i = 0u; i < num_cuboids; i++) {
        result &= !filled(finder, i, mapping[i] >> 2u);
    }

    return result;
}

/// Returns whether two mappings are equal.
fn mapping_eq(a: array<u32, num_cuboids>, b: array<u32, num_cuboids>) -> bool {
    var result = true;
    for (var i = 0u; i < num_cuboids; i++) {
        result &= a[i] == b[i];
    }
    return result;
}

fn covers_surface(finder: ptr<function, Finder>) -> bool {
    var surfaces = (*finder).surfaces;

    // Return early in the common case that there aren't enough potential
    // instructions to possibly fill the remaining squares.
    if num_filled(&surfaces, 0) + (*finder).potential_len < area {
        return false;
    }

    for (var i = 0u; i < (*finder).potential_len; i++) {
        let index = (*finder).potential[i];
        var instruction = (*finder).queue[index];
        if valid(finder, instruction) {
            for (var i = 0u; i < num_cuboids; i++) {
                set_filled_alt(&surfaces, i, instruction.mapping[i] >> 2u, true);
            }
        }
    }

    for (var i = 0u; i < num_cuboids; i++) {
        if num_filled(&surfaces, i) < area {
            return false;
        }
    }

    return true;
}

/// A version of set_filled which takes a list of surfaces instead of a Finder
/// (https://github.com/gfx-rs/wgpu/issues/4540)
fn set_filled_alt(surfaces: ptr<function, array<array<u32, surface_words>, num_cuboids>>, cuboid: u32, square: u32, value: bool) {
    (*surfaces)[cuboid][square / 32] &= ~(1u << (square % 32));
    (*surfaces)[cuboid][square / 32] |= (u32(value) << (square % 32));
}

/// Returns the number of filled squares on a surface.
fn num_filled(surfaces: ptr<function, array<array<u32, surface_words>, num_cuboids>>, i: u32) -> u32 {
    var total = 0u;
    for (var j = 0u; j < surface_words; j++) {
        total += countOneBits((*surfaces)[i][j]);
    }
    return total;
}

/// Converts a `Finder` to a `FinderInfo`.
fn finder_to_info(finder_p: ptr<function, Finder>) -> FinderInfo {
    var finder = *finder_p;
    // Save this before it's overwritten.
    let base_index = finder.base_index;
    var rev_decisions = array<bool, queue_capacity>();
    var rev_decisions_len = 0u;
    var rev_base_decision = -1;

    for (var i = i32(finder.index - 1); i >= 0; i--) {
        if finder.queue[i].followup_index != 0 {
            // The instruction was run: add a decision of 1, and backtrack it so we're in
            // the right state for the next instruction.
            rev_decisions[rev_decisions_len] = true;
            if u32(i) == base_index {
                rev_base_decision = i32(rev_decisions_len);
            }
            rev_decisions_len += 1;

            finder.base_index = 0;
            backtrack(&finder);
        } else {
            // The instruction wasn't run: run it to see if it was invalid or if it was
            // backtracked.
            finder.index = u32(i);
            let old_len = finder.queue_len;
            handle_instruction(&finder);
            if finder.queue_len > old_len {
                // We were able to run it, which means it was valid and was just backtracked at
                // some point: add a decision of 0 then backtrack it again.
                rev_decisions[rev_decisions_len] = false;
                rev_decisions_len += 1;

                finder.base_index = 0;
                backtrack(&finder);
            }
        }
    }

    var decisions = array<u32, decision_words>();
    for (var i = 0u; i < rev_decisions_len; i++) {
        decisions[i / 32] |= u32(rev_decisions[rev_decisions_len - 1 - i]) << (i % 32);
    }

    return FinderInfo(
        finder.queue[0].mapping,
        decisions,
        rev_decisions_len,
        u32(i32(rev_decisions_len) - 1 - rev_base_decision),
    );
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
    maybe_solutions.items[index] = finder_to_info(finder);
    return true;
}
