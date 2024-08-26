from amaranth import *
from amaranth.lib import data
from amaranth.utils import ceil_log2

from .base_types import instruction_layout

FINDERS_PER_CORE = 3


def instruction_ref_layout(max_area: int):
    """Returns the layout of an instruction reference."""

    return data.StructLayout(
        {
            # The index of the instruction's parent in the run stack.
            "parent": ceil_log2(max_area),
            # The index of this instruction in its parent's list of valid children.
            #
            # If the index is past the end of that list, it represents the last valid
            # child. Then we always store the last valid child as 11, so that when
            # backtracking we can immediately see 'oh this is the last one, so we need to
            # move onto the next instruction'.
            "child_index": 2,
        }
    )


def max_potential_len(max_area: int):
    """
    Returns the maximum number of potential instructions there can be at any given
    time.
    """

    # The upper bound of how many potential instructions there can be is if every
    # square on the surfaces, except for the ones set by the first instruction, has
    # 4 potential instructions trying to set it: 1 from each direction.
    #
    # While this isn't actually possible, it's a nice clean upper bound.
    return 4 * (max_area - 1)


def max_decisions_len(max_area: int):
    """Returns the maximum number of decisions there can be at any given time."""

    # There's always 1 decision for the first instruction, then the upper bound is
    # that every square has 4 instructions setting it, 3 of which we decided not to
    # run and the last one we did.
    return 1 + 4 * (max_area - 1)


def run_stack_entry_layout(cuboids: int, max_area: int):
    """Returns the layout of a run stack entry."""

    return data.StructLayout(
        {
            # The instruction that was run.
            "instruction": instruction_layout(cuboids, max_area),
            # A reference to where in the run stack this instruction originally came from.
            "source": instruction_ref_layout(max_area),
            # Whether this instruction's child in each direction was valid at the time this
            # instruction was run.
            "children": 4,
            # The number of potential instructions there were at the point when it was run.
            "potential_len": ceil_log2(max_potential_len(max_area) + 1),
            # The index of the decision to run this instruction in the list of decisions.
            "decision_index": ceil_log2(max_decisions_len(max_area)),
        }
    )
