import enum

from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.fifo import SyncFIFO
from amaranth.lib.memory import Memory, ReadPort, WritePort
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from .base_types import instruction_layout, mapping_layout, next_power_of_two
from .main_pipeline import (
    FINDERS_PER_CORE,
    MainPipeline,
    Task,
    instruction_ref_layout,
    max_decisions_len,
    max_potential_len,
    max_run_stack_len,
    run_stack_entry_layout,
)
from .memory import ChunkedMemory, ChunkedReadPortSignature, ChunkedWritePortSignature
from .neighbour_lookup import neighbour_lookup_layout
from .net import shard_depth
from .skip_checker import undo_lookup_layout
from .utils import sig


def prefix_layout(cuboids: int, max_area: int):
    return data.StructLayout(
        {
            "area": range(max_area + 1),
            "start_mapping": mapping_layout(cuboids, max_area),
            "start_mapping_index": (cuboids - 1) * ceil_log2(max_area),
            # I think this can reach `max_decisions_len`: if you can get to a particular length,
            # it had to end with 1 at some point, at which point you can split and increment
            # the base decision.
            "base_decision": range(max_decisions_len(max_area) + 1),
        }
    )


class FinderType(enum.Enum):
    """The reason a finder is being emitted."""

    # The state the finder's in has passed the initial test for likely having a
    # solution (there's an instruction or potential instruction setting every square
    # on the surfaces).
    Solution = 0

    # The finder is a response to `req_split`.
    Split = 1

    # The finder is a response to `req_pause`.
    Pause = 2


def core_in_layout():
    return data.StructLayout(
        {
            # The bit of the finder being received.
            "data": 1,
            # Whether `data` is the last bit of this finder.
            "last": 1,
        }
    )


def core_out_layout():
    return data.StructLayout(
        {
            # The bit of the finder being sent.
            "data": 1,
            # Whether `data` is the last bit of this finder.
            "last": 1,
            # The reason this finder is being emitted (stays the same for the whole finder).
            "type": FinderType,
        }
    )


class CoreInterface(wiring.Signature):
    def __init__(self, max_area: int):
        super().__init__(
            {
                "sink": In(stream.Signature(core_in_layout())),
                "source": Out(stream.Signature(core_out_layout())),
                "req_pause": In(1),
                "req_split": In(1),
                "wants_finder": Out(1),
                "stepping": Out(1),
                "state": Out(State),
                "base_decision": Out(range(max_decisions_len(max_area) + 1)),
            }
        )


class State(enum.Enum):
    Clear = 0
    Receive = 1
    Run = 2
    Check = 3
    # TODO: seems like someone's solution is getting misinterpreted as a Split. Probably something to do with SafeDemux locking in the wrong thing?
    # I think we fixed this but I forget how, I'm gonna leave this here until I spot it in the diff
    Solution = 4
    Pause = 5
    Split = 6


def wb_if_layout(cuboids: int, max_area: int):
    return data.StructLayout(
        {
            "prev_state": State,
            # The 'normal' instruction that WB stage wanted to run next: so, not the first
            # instruction or a potential instruction.
            #
            # `wb_target.parent` is allowed to be past the end of the run stack: that means
            # that whenever this ends up getting processed, it'll actually result in a
            # backtrack instead of this being run.
            "prev_target": instruction_ref_layout(max_area),
            # Whether WB stage actually ended up processing `wb_target`.
            "prev_target_processed": 1,
            # Like `wb_target`, except that when `wb_target.parent` is past the end of the
            # run stack, this is the instruction that was backtracked.
            "prev_inst_ref": instruction_ref_layout(max_area),
            "prev_potential_index": range(max_potential_len(max_area)),
            "prev_decision_index": range(max_decisions_len(max_area)),
            # The index we're clearing: used both for Clear state and clearing the potential
            # surfaces in the background.
            "prev_clear_index": range(shard_depth(max_area)),
            "prev_prefix_done": 1,
            # The value WB stage read from `decisions`.
            "prev_read_decision": 1,
            # If we're in Check state, whether or not we need to wait for the potential
            # surfaces to finish being cleared before proceeding.
            "prev_clearing": 1,
            # If we're in Split state, whether we've already finished sending the finder and
            # are now just searching for a new `base_decision`.
            "prev_finder_done": 1,
            "prev_received": 1,
            "prev_sent": 1,
            "prev_in_data": core_in_layout(),
            "next_prefix": prefix_layout(cuboids, max_area),
            "next_prefix_bits_left": range(prefix_layout(cuboids, max_area).size + 1),
            "next_run_stack_len": range(max_run_stack_len(max_area) + 1),
            "next_potential_len": range(max_potential_len(max_area) + 1),
            "next_decisions_len": range(max_decisions_len(max_area) + 1),
            "next_potential_areas": data.ArrayLayout(range(max_area + 1), cuboids),
        }
    )


def if_nl_layout(cuboids: int, max_area: int):
    return data.StructLayout(
        {
            "state": State,
            "prefix": prefix_layout(cuboids, max_area),
            "prefix_bits_left": range(prefix_layout(cuboids, max_area).size + 1),
            "run_stack_len": range(max_run_stack_len(max_area) + 1),
            "potential_len": range(max_potential_len(max_area) + 1),
            "decisions_len": range(max_decisions_len(max_area) + 1),
            "potential_areas": data.ArrayLayout(range(max_area + 1), cuboids),
            "target": instruction_ref_layout(max_area),
            "potential_index": range(max_potential_len(max_area)),
            "decision_index": range(max_decisions_len(max_area)),
            "clear_index": range(shard_depth(max_area)),
            "child_index": 2,
            "in_rdy": 1,
            "in_data": core_in_layout(),
            "task": Task,
        }
    )


def nl_vc_layout(cuboids: int, max_area: int):
    return data.StructLayout(
        {
            "state": State,
            "prefix": prefix_layout(cuboids, max_area),
            "prefix_bits_left": range(prefix_layout(cuboids, max_area).size + 1),
            "run_stack_len": range(max_run_stack_len(max_area) + 1),
            "potential_len": range(max_potential_len(max_area) + 1),
            "decisions_len": range(max_decisions_len(max_area) + 1),
            "potential_areas": data.ArrayLayout(range(max_area + 1), cuboids),
            "prefix_done": 1,
            "target": instruction_ref_layout(max_area),
            "potential_index": range(max_potential_len(max_area)),
            "decision_index": range(max_decisions_len(max_area)),
            "clear_index": range(shard_depth(max_area)),
            "child_index": 2,
            "in_rdy": 1,
            "in_data": core_in_layout(),
            "task": Task,
            "entry": run_stack_entry_layout(cuboids, max_area),
        }
    )


def vc_wb_layout(cuboids: int, max_area: int):
    return data.StructLayout(
        {
            "state": State,
            "prefix": prefix_layout(cuboids, max_area),
            "prefix_bits_left": range(prefix_layout(cuboids, max_area).size + 1),
            "run_stack_len": range(max_run_stack_len(max_area) + 1),
            "potential_len": range(max_potential_len(max_area) + 1),
            "decisions_len": range(max_decisions_len(max_area) + 1),
            "potential_areas": data.ArrayLayout(range(max_area + 1), cuboids),
            "prefix_done": 1,
            "target": instruction_ref_layout(max_area),
            "potential_index": range(max_potential_len(max_area)),
            "decision_index": range(max_decisions_len(max_area)),
            "clear_index": range(shard_depth(max_area)),
            "child_index": 2,
            "in_rdy": 1,
            "in_data": core_in_layout(),
            "task": Task,
            "entry": run_stack_entry_layout(cuboids, max_area),
            "instruction": instruction_layout(cuboids, max_area),
            "clearing": 1,
        }
    )


class CoreIf(wiring.Component):
    def __init__(self, cuboids: int, max_area: int):
        super().__init__(
            {
                "i": In(
                    wb_if_layout(cuboids, max_area),
                    init={
                        "next_prefix_bits_left": prefix_layout(cuboids, max_area).size
                    },
                ),
                "o": Out(if_nl_layout(cuboids, max_area)),
                "finder": In(2),
                "req_pause": In(1),
                "req_split": In(1),
                "in_fifo_payload": In(core_in_layout()),
                "in_fifo_valid": In(1),
                "potential_read": In(
                    ChunkedReadPortSignature(
                        chunk_width=ceil_log2(FINDERS_PER_CORE),
                        addr_width=ceil_log2(max_potential_len(max_area)),
                        shape=instruction_ref_layout(max_area),
                    )
                ),
                "run_stack_index": Out(range(max_run_stack_len(max_area))),
            }
        )

        self._cuboids = cuboids
        self._max_area = max_area

    def elaborate(self, platform) -> Module:
        m = Module()

        i = self.i
        o = self.o

        m.d.comb += o.prefix_bits_left.eq(i.next_prefix_bits_left)
        m.d.comb += o.run_stack_len.eq(i.next_run_stack_len)
        m.d.comb += o.potential_len.eq(i.next_potential_len)
        m.d.comb += o.decisions_len.eq(i.next_decisions_len)
        m.d.comb += o.potential_areas.eq(i.next_potential_areas)

        prefix_done = sig(m, o.prefix_bits_left == 0)

        next_target = Signal(instruction_ref_layout(self._max_area))
        # TODO: this is the same as `next_inst + 1` (if we switch around the field
        # order). I think this is clearer, but switch to that if it ends up improving
        # performance.
        with m.If(i.prev_inst_ref.child_index == 3):
            m.d.comb += next_target.parent.eq(i.prev_inst_ref.parent + 1)
            m.d.comb += next_target.child_index.eq(0)
        with m.Else():
            m.d.comb += next_target.parent.eq(i.prev_inst_ref.parent)
            m.d.comb += next_target.child_index.eq(i.prev_inst_ref.child_index + 1)

        # If WB stage processed its target, we can move on to the next one, otherwise we
        # need to keep trying to process `prev_target`.
        m.d.comb += o.target.eq(
            Mux(i.prev_target_processed, next_target, i.prev_target)
        )

        backtrack = sig(
            m, (o.run_stack_len != 0) & (o.target.parent == o.run_stack_len)
        )

        m.d.comb += o.potential_index.eq(
            Mux(
                i.prev_state == State.Check,
                i.prev_potential_index + ~i.prev_clearing,
                0,
            )
        )

        def run_transitions(allow_check: bool = True):
            """
            Rather than just writing `state.eq(State.Run)`, you should invoke this
            function in order to make sure that we don't end up accidentally spending an
            iteration in Run state when we should be doing something else.
            """

            with m.If(self.req_pause):
                m.d.comb += o.state.eq(State.Pause)
            with m.Elif(
                self.req_split & (i.next_prefix.base_decision < o.decisions_len)
            ):
                m.d.comb += o.state.eq(State.Split)
                # Set `base_decision` to what the base decision of the finder we're sending will
                # be (1 past the end of its decisions), so that it gets sent out along with the
                # rest of the prefix.
                #
                # However, it might not be our new base decision, since it might be a 0: we'll
                # fix it up once we find the first 1 past our old base decision.
                m.d.comb += o.prefix.base_decision.eq(i.next_prefix.base_decision + 1)
            if allow_check:
                with m.Elif(
                    backtrack
                    & (o.run_stack_len + o.potential_len >= i.next_prefix.area)
                ):
                    # There are enough run + potential instructions that we might have a solution,
                    # so check for that before we backtrack.
                    m.d.comb += o.state.eq(State.Check)
            with m.Else():
                # Note that this also covers the case where we backtrack immediately.
                m.d.comb += o.state.eq(State.Run)

        decision_sent = sig(m, i.prev_sent & i.prev_prefix_done)

        m.d.comb += o.prefix.eq(i.next_prefix)
        with m.Switch(i.prev_state):
            with m.Case(State.Clear):
                with m.If(i.prev_clear_index == shard_depth(self._max_area) - 1):
                    m.d.comb += o.state.eq(State.Receive)
                with m.Else():
                    m.d.comb += o.state.eq(State.Clear)
            with m.Case(State.Receive):
                with m.If(i.prev_received & i.prev_in_data.last):
                    run_transitions()
                with m.Else():
                    m.d.comb += o.state.eq(State.Receive)
            with m.Case(State.Run):
                run_transitions()
            with m.Case(State.Check):
                all_squares_filled = sig(
                    m,
                    Cat(
                        o.run_stack_len + o.potential_areas[k] == i.next_prefix.area
                        for k in range(self._cuboids)
                    ).all(),
                )
                with m.If(~i.prev_clearing & all_squares_filled):
                    # All the squares are filled, which means we have a potential solution!
                    m.d.comb += o.state.eq(State.Solution)
                # Note: although `wb_potential_index` can only go up to `max_potential_len - 1`,
                # this can go all the way up to `max_potential_len` thanks to Amaranth inferring
                # a shape big enough to fit all possible values of `i.prev_potential_index +
                # ~i.prev_clearing`.
                with m.Elif(o.potential_index == o.potential_len):
                    # We've run all the potential instructions and not all the squares are filled,
                    # so this isn't a solution. Time to backtrack.
                    run_transitions(allow_check=False)
                with m.Else():
                    m.d.comb += o.state.eq(State.Check)
            with m.Case(State.Solution):
                with m.If(
                    (i.prev_finder_done | decision_sent)
                    & (i.prev_decision_index == o.decisions_len - 1)
                ):
                    run_transitions(allow_check=False)
                with m.Else():
                    m.d.comb += o.state.eq(State.Solution)
            with m.Case(State.Pause):
                # We transition out of `State.Pause` via. `local_reset`, rather than via. a
                # regular state transition.
                m.d.comb += o.state.eq(State.Pause)
            with m.Case(State.Split):
                # We don't actually stop once the finder is sent like you might expect: since
                # `base_decision` can't point to a 0, we have to keep going until we find a 1 to
                # set it to, or hit the end of `decisions` if there aren't any.
                with m.If(
                    (i.prev_finder_done & i.prev_read_decision)
                    | (
                        (decision_sent | i.prev_finder_done)
                        & (i.prev_decision_index == o.decisions_len - 1)
                    )
                ):
                    run_transitions()
                with m.Else():
                    m.d.comb += o.state.eq(State.Split)

        # We only get to Clear state via. resetting, which will reset this to 0 anyway:
        # so the only time we actually need to reset it is when exiting Check state so
        # that we don't waste time clearing addresses the potential surfaces don't have.
        m.d.comb += o.clear_index.eq(
            Mux(
                (i.prev_state == State.Check) & (o.state != State.Check),
                0,
                i.prev_clear_index + 1,
            )
        )

        # Note: technically this is still incorrect, since it'd be possible to go
        # directly from one split into another if `req_split` was already asserted when
        # finishing the first one. That shouldn't happen in practice since our SoC won't
        # ask anyone to split while there's already a split happening, but we might
        # still want to fix it at some point.
        with m.If(o.state != i.prev_state):
            m.d.comb += o.decision_index.eq(0)
        with m.Else():
            m.d.comb += o.decision_index.eq(
                i.prev_decision_index
                + (decision_sent | ((o.state == State.Split) & i.prev_finder_done))
            )

        m.d.comb += self.potential_read.chunk.eq(self.finder)
        m.d.comb += self.potential_read.addr.eq(o.potential_index)

        with m.If(o.state == State.Check):
            m.d.comb += self.run_stack_index.eq(self.potential_read.data.parent)
        with m.Elif(backtrack):
            m.d.comb += self.run_stack_index.eq(o.run_stack_len - 1)
        with m.Else():
            m.d.comb += self.run_stack_index.eq(o.target.parent)

        m.d.comb += o.child_index.eq(
            Mux(
                o.state == State.Check,
                self.potential_read.data.child_index,
                o.target.child_index,
            )
        )

        m.d.comb += o.in_rdy.eq(self.in_fifo_valid)
        m.d.comb += o.in_data.eq(self.in_fifo_payload)

        with m.Switch(o.state):
            with m.Case(State.Clear):
                m.d.comb += o.task.eq(Task.Clear)
            with m.Case(State.Receive):
                m.d.comb += o.task.eq(
                    # If we've received a decision of 1, we need to run the next valid instruction
                    # to fulfil it; otherwise, we want to not run the next valid instruction, but we
                    # still need to check whether it was valid so that we know whether we can move
                    # on to the next decision.
                    Mux(
                        prefix_done & o.in_rdy & o.in_data.data,
                        Task.Advance,
                        # This also serves as a no-op in the case where there isn't another bit
                        # available yet.
                        Task.Check,
                    )
                )
            with m.Case(State.Run):
                m.d.comb += o.task.eq(Mux(backtrack, Task.Backtrack, Task.Advance))
            with m.Case(State.Check):
                m.d.comb += o.task.eq(Task.Check)
            with m.Default():
                # In states that don't need the main pipeline, give it the Check task, since it
                # doesn't have any side effects.
                m.d.comb += o.task.eq(Task.Check)
        return m


class CoreNl(wiring.Component):
    def __init__(self, cuboids: int, max_area: int):
        super().__init__(
            {
                "i": In(
                    if_nl_layout(cuboids, max_area),
                    init={"prefix_bits_left": prefix_layout(cuboids, max_area).size},
                ),
                "o": Out(nl_vc_layout(cuboids, max_area)),
                "finder": In(2),
                "entry": In(run_stack_entry_layout(cuboids, max_area)),
                # Inputs to the `MainPipeline` (which don't happen to already be included in
                # `o`).
                "pipeline_entry": Out(run_stack_entry_layout(cuboids, max_area)),
                "pipeline_child": Out(1),
            }
        )

        self._cuboids = cuboids
        self._max_area = max_area

    def elaborate(self, platform) -> Module:
        m = Module()

        i = self.i
        o = self.o

        local_reset = sig(
            m,
            (
                (i.state == State.Pause)
                & (i.prefix_bits_left == 0)
                & (i.decision_index == i.decisions_len)
            )
            | (
                (i.task == Task.Backtrack)
                # TODO: hey, we could figure this out sooner if we made base_decision an index in the run stack instead.
                # more importantly, we could avoid having to scan for the next 1 when splitting.
                #
                # hm, i guess it's not quite that simple since the base_decision is used to figure out which decision we can split on.
                # you can solve that by just adding a counter of the number of 1s seen, though.
                & (self.entry.decision_index < i.prefix.base_decision)
            ),
        )

        m.d.comb += o.state.eq(Mux(local_reset, State.Clear, i.state))
        m.d.comb += o.prefix.eq(i.prefix)
        m.d.comb += o.prefix_bits_left.eq(
            Mux(local_reset, o.prefix.shape().size, i.prefix_bits_left)
        )
        m.d.comb += o.run_stack_len.eq(Mux(local_reset, 0, i.run_stack_len))
        m.d.comb += o.potential_len.eq(Mux(local_reset, 0, i.potential_len))
        m.d.comb += o.decisions_len.eq(Mux(local_reset, 0, i.decisions_len))
        m.d.comb += o.potential_areas.eq(Mux(local_reset, 0, i.potential_areas))
        m.d.comb += o.prefix_done.eq(o.prefix_bits_left == 0)
        m.d.comb += o.target.eq(Mux(local_reset, 0, i.target))
        m.d.comb += o.potential_index.eq(i.potential_index)
        m.d.comb += o.decision_index.eq(i.decision_index)
        m.d.comb += o.clear_index.eq(Mux(local_reset, 0, i.clear_index))
        m.d.comb += o.child_index.eq(i.child_index)
        m.d.comb += o.in_rdy.eq(i.in_rdy)
        m.d.comb += o.in_data.eq(i.in_data)
        m.d.comb += o.task.eq(Mux(local_reset, Task.Clear, i.task))
        m.d.comb += o.entry.eq(self.entry)

        m.d.comb += self.pipeline_entry.eq(o.entry)
        with m.If(o.run_stack_len == 0):
            m.d.comb += self.pipeline_entry.instruction.pos.x.eq(0)
            m.d.comb += self.pipeline_entry.instruction.pos.y.eq(0)
            m.d.comb += self.pipeline_entry.instruction.mapping.eq(
                o.prefix.start_mapping
            )
        m.d.comb += self.pipeline_child.eq(
            (o.task != Task.Backtrack) & (o.run_stack_len != 0)
        )

        return m


class CoreVc(wiring.Component):
    def __init__(self, cuboids: int, max_area: int):
        super().__init__(
            {
                "i": In(
                    nl_vc_layout(cuboids, max_area),
                    init={"prefix_bits_left": prefix_layout(cuboids, max_area).size},
                ),
                "o": Out(vc_wb_layout(cuboids, max_area)),
                "finder": In(2),
                "instruction": In(instruction_layout(cuboids, max_area)),
                "potential_surface_indices": Out(range(max_area)).array(cuboids),
                "decisions_index": Out(range(max_decisions_len(max_area))),
            }
        )

        self._cuboids = cuboids
        self._max_area = max_area

    def elaborate(self, platform) -> Module:
        m = Module()

        i = self.i
        o = self.o

        m.d.comb += o.state.eq(i.state)
        m.d.comb += o.prefix.eq(i.prefix)
        m.d.comb += o.prefix_bits_left.eq(i.prefix_bits_left)
        m.d.comb += o.run_stack_len.eq(i.run_stack_len)
        m.d.comb += o.potential_len.eq(i.potential_len)
        m.d.comb += o.decisions_len.eq(i.decisions_len)
        m.d.comb += o.potential_areas.eq(i.potential_areas)
        m.d.comb += o.prefix_done.eq(i.prefix_done)
        m.d.comb += o.target.eq(i.target)
        m.d.comb += o.potential_index.eq(i.potential_index)
        m.d.comb += o.decision_index.eq(i.decision_index)
        m.d.comb += o.clear_index.eq(i.clear_index)
        m.d.comb += o.in_rdy.eq(i.in_rdy)
        m.d.comb += o.in_data.eq(i.in_data)
        m.d.comb += o.task.eq(i.task)
        m.d.comb += o.entry.eq(i.entry)

        m.d.comb += o.instruction.eq(self.instruction)

        # The `vc_potential_index == 0` is necessary because otherwise Check state would
        # freeze up as soon as it ran the first potential instruction and the potential
        # surfaces weren't empty anymore.
        m.d.comb += o.clearing.eq(
            (o.state != State.Check)
            | (
                (o.potential_index == 0)
                & Cat(o.potential_areas[k] != 0 for k in range(self._cuboids)).any()
            ),
        )

        for i in range(self._cuboids):
            m.d.comb += self.potential_surface_indices[i].eq(
                Mux(
                    o.clearing,
                    o.clear_index,
                    o.instruction.mapping[i].square,
                )
            )

        m.d.comb += self.decisions_index.eq(o.decision_index)

        return m


class CoreWb(wiring.Component):
    def __init__(self, cuboids: int, max_area: int):
        super().__init__(
            {
                "i": In(
                    vc_wb_layout(cuboids, max_area),
                    init={
                        "prefix_bits_left": prefix_layout(cuboids, max_area).size,
                        "clearing": 1,
                    },
                ),
                "o": Out(wb_if_layout(cuboids, max_area)),
                "finder": In(2),
                "instruction_valid": In(1),
                "neighbours_valid": In(4),
                "decisions_data": In(1),
                "potential_surface_data": In(3),
                "in_fifo_ready": Out(1),
                # Use the FIFO's names rather than the stream names, since we don't actually
                # follow the stream protocol (valid depends on ready)
                "out_fifo_rdy": In(1),
                "out_fifo_en": Out(1),
                "out_fifo_data": Out(core_out_layout()),
                "stepping": Out(1),
                "run_stack_write": In(
                    WritePort.Signature(
                        addr_width=ceil_log2(FINDERS_PER_CORE)
                        + ceil_log2(max_run_stack_len(max_area)),
                        shape=run_stack_entry_layout(cuboids, max_area),
                    )
                ),
                "potential_write": In(
                    ChunkedWritePortSignature(
                        chunk_width=ceil_log2(FINDERS_PER_CORE),
                        addr_width=ceil_log2(max_potential_len(max_area)),
                        shape=instruction_ref_layout(max_area),
                    )
                ),
                "decisions_write": In(
                    ChunkedWritePortSignature(
                        chunk_width=ceil_log2(FINDERS_PER_CORE),
                        addr_width=ceil_log2(max_decisions_len(max_area)),
                        shape=1,
                    )
                ),
                "potential_surface_writes": In(
                    ChunkedWritePortSignature(
                        chunk_width=ceil_log2(FINDERS_PER_CORE),
                        addr_width=ceil_log2(max_area),
                        shape=1,
                    )
                ).array(3),
            }
        )

        self._cuboids = cuboids
        self._max_area = max_area

    def elaborate(self, platform) -> Module:
        m = Module()

        i = self.i
        o = self.o

        m.d.comb += o.prev_state.eq(i.state)
        m.d.comb += o.prev_prefix_done.eq(i.prefix_done)
        m.d.comb += o.prev_target.eq(i.target)
        m.d.comb += o.prev_potential_index.eq(i.potential_index)
        m.d.comb += o.prev_decision_index.eq(i.decision_index)
        m.d.comb += o.prev_clear_index.eq(i.clear_index)
        m.d.comb += o.prev_in_data.eq(i.in_data)
        m.d.comb += o.prev_clearing.eq(i.clearing)

        m.d.comb += o.prev_read_decision.eq(self.decisions_data)

        run = sig(
            m,
            (i.task == Task.Advance)
            & self.instruction_valid
            & self.neighbours_valid.any(),
        )
        # Potential instructions still need to be added in Receive state even if the
        # next decision is 0.
        potential = sig(
            m,
            ((i.task == Task.Advance) | ((i.state == State.Receive) & i.prefix_done))
            & self.instruction_valid
            & ~self.neighbours_valid.any(),
        )

        received = sig(
            m,
            (i.state == State.Receive)
            & i.in_rdy
            # If the instruction wasn't valid, whether or not to run it wasn't a decision.
            & ~(
                i.prefix_done & (~self.instruction_valid | ~self.neighbours_valid.any())
            ),
        )
        m.d.comb += o.prev_received.eq(received)
        m.d.comb += self.in_fifo_ready.eq(received)

        # The `i.prefix_done` is needed so that we don't try and interpret the meaning
        # of `i.prefix.base_decision` while the prefix is still being shifted out.
        finder_done = sig(
            m, i.prefix_done & (i.decision_index >= i.prefix.base_decision)
        )
        m.d.comb += o.prev_finder_done.eq(finder_done)

        sent = sig(
            m,
            (
                (i.state == State.Solution)
                | (i.state == State.Pause)
                | ((i.state == State.Split) & ~finder_done)
            )
            & self.out_fifo_rdy,
        )
        m.d.comb += o.prev_sent.eq(sent)

        split_reached = sig(
            m,
            (i.state == State.Split) & (i.decision_index == i.prefix.base_decision - 1),
        )
        m.d.comb += self.out_fifo_data.data.eq(
            Mux(
                i.prefix_done,
                self.decisions_data & ~split_reached,
                i.prefix.as_value()[-1],
            )
        )
        m.d.comb += self.out_fifo_data.last.eq(
            Mux(
                i.prefix_done,
                (i.decision_index == i.decisions_len - 1) | split_reached,
                (i.decisions_len == 0) & (i.prefix_bits_left == 1),
            )
        )
        with m.Switch(i.state):
            with m.Case(State.Solution):
                m.d.comb += self.out_fifo_data.type.eq(FinderType.Solution)
            with m.Case(State.Pause):
                m.d.comb += self.out_fifo_data.type.eq(FinderType.Pause)
            with m.Case(State.Split):
                m.d.comb += self.out_fifo_data.type.eq(FinderType.Split)
            # It doesn't really matter what this is in other states, leave it as 0.

        m.d.comb += self.out_fifo_en.eq(sent)
        m.d.comb += self.stepping.eq(run | (i.task == Task.Backtrack))

        last_child = sig(m, i.target.child_index == sum(i.entry.children) - 1)
        normalised_target = Signal.like(i.target)
        m.d.comb += normalised_target.parent.eq(i.target.parent)
        m.d.comb += normalised_target.child_index.eq(
            Mux(last_child, 3, i.target.child_index)
        )

        m.d.comb += self.run_stack_write.addr.eq(Cat(i.run_stack_len, self.finder))
        m.d.comb += self.run_stack_write.data.instruction.eq(i.instruction)
        m.d.comb += self.run_stack_write.data.source.eq(normalised_target)
        m.d.comb += self.run_stack_write.data.children.eq(self.neighbours_valid)
        m.d.comb += self.run_stack_write.data.potential_len.eq(i.potential_len)
        m.d.comb += self.run_stack_write.data.decision_index.eq(i.decisions_len)
        m.d.comb += self.run_stack_write.en.eq(run)

        m.d.comb += self.potential_write.chunk.eq(self.finder)
        m.d.comb += self.potential_write.addr.eq(i.potential_len)
        m.d.comb += self.potential_write.data.eq(i.target)
        m.d.comb += self.potential_write.en.eq(potential)

        m.d.comb += self.decisions_write.chunk.eq(self.finder)
        m.d.comb += self.decisions_write.addr.eq(
            Mux(i.task == Task.Backtrack, i.entry.decision_index, i.decisions_len)
        )
        m.d.comb += self.decisions_write.data.eq(
            Mux(i.state == State.Receive, i.in_data.data, i.task != Task.Backtrack)
        )
        m.d.comb += self.decisions_write.en.eq(
            (received & i.prefix_done) | run | (i.task == Task.Backtrack)
        )

        for k in range(self._cuboids):
            write_port = self.potential_surface_writes[k]
            m.d.comb += write_port.chunk.eq(self.finder)
            m.d.comb += write_port.addr.eq(
                Mux(
                    i.clearing,
                    i.clear_index,
                    i.instruction.mapping[k].square,
                )
            )
            m.d.comb += write_port.data.eq(~i.clearing)
            m.d.comb += write_port.en.eq(
                Mux(
                    i.clearing,
                    i.clear_index < self._max_area,
                    # TODO: could this cause issues due to us assuming Task.Check has no side effects?
                    # no, I think it should always get safely cleared, but that still sucks.
                    self.instruction_valid,
                )
            )

        shift_prefix = sig(m, (received | sent) & ~i.prefix_done)
        prefix_in = sig(
            m, Mux(i.state == State.Receive, i.in_data.data, i.prefix.as_value()[-1])
        )
        m.d.comb += o.next_prefix.eq(
            Mux(shift_prefix, Cat(prefix_in, i.prefix.as_value()[:-1]), i.prefix)
        )
        m.d.comb += o.next_prefix_bits_left.eq(
            Mux(
                ((i.state == State.Receive) & ~(i.in_data.last & received))
                | ((i.state == State.Solution) & ~(self.out_fifo_data.last & sent))
                | (i.state == State.Pause)
                | (
                    (i.state == State.Split)
                    & ~(
                        (finder_done & self.decisions_data)
                        | (
                            (finder_done | (i.prefix_done & sent))
                            & (i.decision_index == i.decisions_len - 1)
                        )
                    )
                ),
                i.prefix_bits_left - shift_prefix,
                i.prefix.shape().size,
            )
        )

        with m.If(
            (i.task == Task.Backtrack)
            & (i.entry.decision_index == i.prefix.base_decision)
        ):
            m.d.comb += o.next_prefix.base_decision.eq(i.prefix.base_decision + 1)
        with m.Elif((i.state == State.Split) & finder_done & self.decisions_data):
            m.d.comb += o.next_prefix.base_decision.eq(i.decision_index)
        with m.Elif(
            (i.state == State.Split)
            # Unlike the previous case, this can actually occur on the same cycle as
            # out.last is set, so just finder_done wouldn't work here.
            & (finder_done | (i.prefix_done & sent))
            & (i.decision_index == i.decisions_len - 1)
        ):
            m.d.comb += o.next_prefix.base_decision.eq(i.decisions_len)

        with m.If(i.task == Task.Backtrack):
            m.d.comb += o.next_run_stack_len.eq(i.run_stack_len - 1)
            m.d.comb += o.next_potential_len.eq(i.entry.potential_len)
        with m.Else():
            m.d.comb += o.next_run_stack_len.eq(i.run_stack_len + run)
            m.d.comb += o.next_potential_len.eq(i.potential_len + potential)

        with m.Switch(i.state):
            with m.Case(State.Receive):
                m.d.comb += o.next_decisions_len.eq(
                    i.decisions_len + (received & i.prefix_done)
                )
            with m.Case(State.Run):
                with m.If(i.task == Task.Backtrack):
                    m.d.comb += o.next_decisions_len.eq(i.entry.decision_index + 1)
                with m.Else():
                    m.d.comb += o.next_decisions_len.eq(i.decisions_len + run)
            with m.Default():
                m.d.comb += o.next_decisions_len.eq(i.decisions_len)

        for k in range(self._cuboids):
            read_data = self.potential_surface_data[k]
            write_port = self.potential_surface_writes[k]
            with m.If(i.state == State.Clear):
                m.d.comb += o.next_potential_areas[k].eq(i.potential_areas[k])
            with m.Elif(write_port.en & (read_data == 0) & (write_port.data == 1)):
                m.d.comb += o.next_potential_areas[k].eq(i.potential_areas[k] + 1)
            with m.Elif(write_port.en & (read_data == 1) & (write_port.data == 0)):
                m.d.comb += o.next_potential_areas[k].eq(i.potential_areas[k] - 1)
            with m.Else():
                m.d.comb += o.next_potential_areas[k].eq(i.potential_areas[k])

        m.d.comb += o.prev_target_processed.eq(
            (
                ((i.state == State.Receive) & i.prefix_done & i.in_rdy)
                | (i.state == State.Run)
            )
            & (i.run_stack_len != 0)
        )

        # Arguably we should consider Check state here too, but this is only used for
        # computing the next target anyway so it doesn't really matter.
        #
        # We use `normalised_target` here so that we can always find the next target by
        # just adding 1 to `child_index`.
        m.d.comb += o.prev_inst_ref.eq(
            Mux(i.task == Task.Backtrack, i.entry.source, normalised_target)
        )

        return m


class Core(wiring.Component):
    def __init__(self, cuboids: int, max_area: int):
        nl_layout = neighbour_lookup_layout(max_area)
        ul_layout = undo_lookup_layout(max_area)

        super().__init__(
            {
                "interfaces": Out(CoreInterface(max_area)).array(FINDERS_PER_CORE),
                # The ports this core should use to access the neighbour lookups.
                "neighbour_lookups": In(
                    ReadPort.Signature(
                        addr_width=ceil_log2(nl_layout.depth),
                        shape=nl_layout.shape,
                    )
                ).array(cuboids),
                # The ports this core should use to access the undo lookups.
                "undo_lookups": In(
                    ReadPort.Signature(
                        addr_width=ceil_log2(ul_layout.depth),
                        shape=ul_layout.shape,
                    )
                ).array(cuboids - 1),
                # The state that the finder in WB stage is in.
                #
                # It doesn't really matter which stage this comes from, though: the point is
                # just to find out what state most of the core's time is being spent in, and all
                # the finders will pass through WB stage once per iteration.
                #
                # I chose WB stage because its state comes out of a register, so we don't have
                # to worry about critical paths extending into the core.
                "state": Out(State),
            }
        )

        self._cuboids = cuboids
        self._max_area = max_area

    def elaborate(self, platform) -> Module:
        m = Module()

        run_stack = Memory(
            shape=run_stack_entry_layout(self._cuboids, self._max_area),
            depth=FINDERS_PER_CORE
            * next_power_of_two(max_run_stack_len(self._max_area)),
            init=[],
        )
        m.submodules.run_stack = run_stack

        potential = ChunkedMemory(
            shape=instruction_ref_layout(self._max_area),
            depth=max_potential_len(self._max_area),
            chunks=FINDERS_PER_CORE,
        )
        m.submodules.potential = potential
        potential_read, potential_write = potential.sdp_port(read_domain="comb")

        decisions = ChunkedMemory(
            shape=1, depth=max_decisions_len(self._max_area), chunks=FINDERS_PER_CORE
        )
        m.submodules.decisions = decisions
        decisions_read, decisions_write = decisions.sdp_port()

        potential_surface_ports = []
        for k in range(self._cuboids):
            surface = ChunkedMemory(
                shape=1, depth=self._max_area, chunks=FINDERS_PER_CORE
            )
            m.submodules[f"potential_surface_{k}"] = surface
            potential_surface_ports.append(surface.sdp_port())

        in_fifos = []
        out_fifos = []

        for k in range(FINDERS_PER_CORE):
            # TODO: we don't actually need this, since valid is guaranteed to stay 1 once
            # it's been asserted anyway and it's fine for ready to change willy-nilly.
            in_fifo = SyncFIFO(width=2, depth=1)
            m.submodules[f"in_fifo_{k}"] = in_fifo

            wiring.connect(m, wiring.flipped(self.interfaces[k].sink), in_fifo.w_stream)
            in_fifos.append(in_fifo)

            # TODO: maybe replace this with PipeValid/PipeReady? It wouldn't have any effect
            # on throughput like it would for the input.
            out_fifo = SyncFIFO(width=4, depth=1)
            m.submodules[f"out_fifo_{k}"] = out_fifo

            wiring.connect(
                m, out_fifo.r_stream, wiring.flipped(self.interfaces[k].source)
            )
            out_fifos.append(out_fifo)

        # Where we are in the cycle of finders moving through different pipeline stages.
        #
        # More concretely, the pipeline stage finder 0 is in.
        finder_offset = Signal(range(FINDERS_PER_CORE))
        with m.If(finder_offset == FINDERS_PER_CORE - 1):
            m.d.sync += finder_offset.eq(0)
        with m.Else():
            m.d.sync += finder_offset.eq(finder_offset + 1)

        if_ = CoreIf(self._cuboids, self._max_area)
        nl = CoreNl(self._cuboids, self._max_area)
        vc = CoreVc(self._cuboids, self._max_area)
        wb = CoreWb(self._cuboids, self._max_area)
        m.submodules["if"] = if_
        m.submodules.nl = nl
        m.submodules.vc = vc
        m.submodules.wb = wb

        m.d.sync += if_.i.eq(wb.o)
        m.d.sync += nl.i.eq(if_.o)
        m.d.sync += vc.i.eq(nl.o)
        m.d.sync += wb.i.eq(vc.o)

        # IF
        #
        # I considered making this pipeline stage run at the same time as WB stage, to
        # reduce the amount of finders per core and hence resources used; but doing it
        # that way would require implementing manual forwarding of potential
        # instructions being written in WB stage, and while that wouldn't be
        # particularly hard, merging the two pipeline stages together would be premature
        # optimisation and so we shouldn't do it if it'll make the code worse.
        #
        # In addition, a merged WB/IF stage could very well end up being the critical
        # path of the design, so it's not exactly as though merging them would be a
        # guaranteed win - we may well have ended up having to split them up later
        # anyway.
        m.d.comb += if_.finder.eq((FINDERS_PER_CORE - finder_offset) % FINDERS_PER_CORE)
        m.d.comb += if_.req_pause.eq(Array(self.interfaces)[if_.finder].req_pause)
        m.d.comb += if_.req_split.eq(Array(self.interfaces)[if_.finder].req_split)
        m.d.comb += if_.in_fifo_payload.eq(Array(in_fifos)[if_.finder].r_data)
        m.d.comb += if_.in_fifo_valid.eq(Array(in_fifos)[if_.finder].r_rdy)
        wiring.connect(m, if_.potential_read, potential_read)
        run_stack_read = run_stack.read_port()
        m.d.comb += run_stack_read.addr.eq(Cat(if_.run_stack_index, if_.finder))

        # NL
        m.d.comb += nl.finder.eq(
            (FINDERS_PER_CORE + 1 - finder_offset) % FINDERS_PER_CORE
        )
        m.d.comb += nl.entry.eq(run_stack_read.data)

        main_pipeline = MainPipeline(self._cuboids, self._max_area)
        m.submodules.main_pipeline = main_pipeline

        m.d.comb += main_pipeline.finder.eq(nl.finder)
        m.d.comb += main_pipeline.start_mapping_index.eq(
            nl.o.prefix.start_mapping_index
        )
        m.d.comb += main_pipeline.task.eq(nl.o.task)
        m.d.comb += main_pipeline.entry.eq(nl.pipeline_entry)
        m.d.comb += main_pipeline.child.eq(nl.pipeline_child)
        m.d.comb += main_pipeline.child_index.eq(nl.o.child_index)
        m.d.comb += main_pipeline.clear_index.eq(nl.o.clear_index)
        for k in range(self._cuboids):
            wiring.connect(
                m,
                main_pipeline.neighbour_lookups[k],
                wiring.flipped(self.neighbour_lookups[k]),
            )
        for k in range(self._cuboids - 1):
            wiring.connect(
                m,
                main_pipeline.undo_lookups[k],
                wiring.flipped(self.undo_lookups[k]),
            )

        # VC
        m.d.comb += vc.finder.eq(
            (FINDERS_PER_CORE + 2 - finder_offset) % FINDERS_PER_CORE
        )
        m.d.comb += vc.instruction.eq(main_pipeline.instruction)

        for k in range(self._cuboids):
            read_port, _ = potential_surface_ports[k]
            m.d.comb += read_port.chunk.eq(vc.finder)
            m.d.comb += read_port.addr.eq(vc.potential_surface_indices[k])

        m.d.comb += decisions_read.chunk.eq(vc.finder)
        m.d.comb += decisions_read.addr.eq(vc.decisions_index)

        # WB
        m.d.comb += wb.finder.eq(FINDERS_PER_CORE - 1 - finder_offset)
        m.d.comb += wb.instruction_valid.eq(main_pipeline.instruction_valid)
        m.d.comb += wb.neighbours_valid.eq(main_pipeline.neighbours_valid)
        m.d.comb += wb.decisions_data.eq(decisions_read.data)
        m.d.comb += wb.out_fifo_rdy.eq(Array(out_fifos)[wb.finder].w_rdy)
        wiring.connect(m, wb.run_stack_write, run_stack.write_port())
        wiring.connect(m, wb.potential_write, potential_write)
        wiring.connect(m, wb.decisions_write, decisions_write)
        for k in range(self._cuboids):
            read_port, write_port = potential_surface_ports[k]
            m.d.comb += wb.potential_surface_data[k].eq(read_port.data)
            wiring.connect(m, wb.potential_surface_writes[k], write_port)

        for k in range(FINDERS_PER_CORE):
            m.d.comb += in_fifos[k].r_en.eq((wb.finder == k) & wb.in_fifo_ready)
            m.d.comb += out_fifos[k].w_data.eq(wb.out_fifo_data)
            m.d.comb += out_fifos[k].w_en.eq((wb.finder == k) & wb.out_fifo_en)

            m.d.comb += self.interfaces[k].stepping.eq((wb.finder == k) & wb.stepping)

        for k in range(FINDERS_PER_CORE):
            state = Signal(State)
            prefix = Signal.like(if_.o.prefix)
            prefix_bits_left = Signal.like(if_.o.prefix_bits_left)
            with m.Switch((finder_offset + k) % FINDERS_PER_CORE):
                with m.Case(0):
                    m.d.comb += state.eq(if_.o.state)
                    m.d.comb += prefix.eq(if_.o.prefix)
                    m.d.comb += prefix_bits_left.eq(if_.o.prefix_bits_left)
                with m.Case(1):
                    m.d.comb += state.eq(nl.o.state)
                    m.d.comb += prefix.eq(nl.o.prefix)
                    m.d.comb += prefix_bits_left.eq(nl.o.prefix_bits_left)
                with m.Case(2):
                    m.d.comb += state.eq(vc.o.state)
                    m.d.comb += prefix.eq(vc.o.prefix)
                    m.d.comb += prefix_bits_left.eq(vc.o.prefix_bits_left)
                with m.Case(3):
                    m.d.comb += state.eq(wb.i.state)
                    m.d.comb += prefix.eq(wb.i.prefix)
                    m.d.comb += prefix_bits_left.eq(wb.i.prefix_bits_left)

            m.d.comb += self.interfaces[k].wants_finder.eq(
                (state == State.Receive) & (prefix_bits_left == prefix.shape().size)
            )
            m.d.comb += self.interfaces[k].state.eq(state)
            m.d.comb += self.interfaces[k].base_decision.eq(prefix.base_decision)

        m.d.comb += self.state.eq(wb.i.state)

        return m


if __name__ == "__main__":
    from amaranth.back import verilog

    print(verilog.convert(Core(3, 64)))
