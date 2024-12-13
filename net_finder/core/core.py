import enum

from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.fifo import SyncFIFO
from amaranth.lib.memory import Memory, ReadPort
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from .base_types import mapping_layout, next_power_of_two
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
from .memory import ChunkedMemory
from .neighbour_lookup import neighbour_lookup_layout
from .net import shard_depth
from .skip_checker import undo_lookup_layout
from .utils import pipe


def prefix_layout(cuboids: int, max_area: int):
    return data.StructLayout(
        {
            "area": range(max_area + 1),
            "start_mapping": mapping_layout(cuboids, max_area),
            "start_mapping_index": (cuboids - 1) * ceil_log2(max_area),
            # I think this reach `max_decisions_len`: if you can get to a particular length,
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
        for i in range(self._cuboids):
            surface = ChunkedMemory(
                shape=1, depth=self._max_area, chunks=FINDERS_PER_CORE
            )
            m.submodules[f"potential_surface_{i}"] = surface
            potential_surface_ports.append(surface.sdp_port())

        in_fifos = []
        out_fifos = []

        for i in range(FINDERS_PER_CORE):
            # TODO: we don't actually need this, since valid is guaranteed to stay 1 once
            # it's been asserted anyway and it's fine for ready to change willy-nilly.
            in_fifo = SyncFIFO(width=2, depth=1)
            m.submodules[f"in_fifo_{i}"] = in_fifo

            wiring.connect(m, wiring.flipped(self.interfaces[i].sink), in_fifo.w_stream)
            in_fifos.append(in_fifo)

            # TODO: maybe replace this with PipeValid/PipeReady? It wouldn't have any effect
            # on throughput like it would for the input.
            out_fifo = SyncFIFO(width=4, depth=1)
            m.submodules[f"out_fifo_{i}"] = out_fifo

            wiring.connect(
                m, out_fifo.r_stream, wiring.flipped(self.interfaces[i].source)
            )
            out_fifos.append(out_fifo)

        wb_state = Signal(State)
        # The 'normal' instruction that WB stage wanted to run next: so, not the first
        # instruction or a potential instruction.
        #
        # `wb_target.parent` is allowed to be past the end of the run stack: that means
        # that whenever this ends up getting processed, it'll actually result in a
        # backtrack instead of this being run.
        wb_target = Signal(instruction_ref_layout(self._max_area))
        # Whether WB stage actually ended up processing `wb_target`.
        wb_target_processed = Signal(1)
        # Like `wb_target`, except that when `wb_target.parent` is past the end of the
        # run stack, this is the instruction that was backtracked.
        wb_inst_ref = Signal(instruction_ref_layout(self._max_area))
        wb_potential_index = Signal(range(max_potential_len(self._max_area)))
        wb_decision_index = Signal(range(max_decisions_len(self._max_area)))
        # The index we're clearing: used both for Clear state and clearing the potential
        # surfaces in the background.
        wb_clear_index = Signal(range(shard_depth(self._max_area)))
        wb_prefix_done = Signal(1)

        # The value WB stage read from `decisions`.
        wb_read_decision = Signal(1)
        # If we're in Check state, whether or not we need to wait for the potential
        # surfaces to finish being cleared before proceeding.
        wb_clearing = Signal(1, init=1)
        # If we're in Split state, whether we've already finished sending the finder and
        # are now just searching for a new `base_decision`.
        wb_finder_done = Signal(1)
        wb_received = Signal(1)
        wb_sent = Signal(1)
        wb_in = Signal(core_in_layout())

        wb_next_prefix = Signal(prefix_layout(self._cuboids, self._max_area))
        wb_next_prefix_bits_left = Signal(range(wb_next_prefix.shape().size + 1))
        wb_next_run_stack_len = Signal(range(max_run_stack_len(self._max_area) + 1))
        wb_next_potential_len = Signal(range(max_potential_len(self._max_area) + 1))
        wb_next_decisions_len = Signal(range(max_decisions_len(self._max_area) + 1))
        wb_next_potential_areas = [
            Signal(range(self._max_area + 1)) for i in range(self._cuboids)
        ]

        # Where we are in the cycle of finders moving through different pipeline stages.
        #
        # More concretely, the pipeline stage finder 0 is in.
        finder_offset = Signal(range(FINDERS_PER_CORE))
        with m.If(finder_offset == FINDERS_PER_CORE - 1):
            m.d.sync += finder_offset.eq(0)
        with m.Else():
            m.d.sync += finder_offset.eq(finder_offset + 1)

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

        if_finder = (FINDERS_PER_CORE - finder_offset) % FINDERS_PER_CORE

        if_prev_state = pipe(m, wb_state)
        if_prev_target = pipe(m, wb_target)
        if_prev_target_processed = pipe(m, wb_target_processed)
        if_prev_inst_ref = pipe(m, wb_inst_ref)
        if_prev_potential_index = pipe(m, wb_potential_index)
        if_prev_decision_index = pipe(m, wb_decision_index)
        if_prev_clear_index = pipe(m, wb_clear_index)
        if_prev_prefix_done = pipe(m, wb_prefix_done)
        if_prev_read_decision = pipe(m, wb_read_decision)
        if_prev_clearing = pipe(m, wb_clearing)
        if_prev_finder_done = pipe(m, wb_finder_done)
        if_prev_received = pipe(m, wb_received)
        if_prev_sent = pipe(m, wb_sent)
        if_prev_in = pipe(m, wb_in)

        if_initial_prefix = pipe(m, wb_next_prefix)
        if_prefix_bits_left = pipe(
            m, wb_next_prefix_bits_left, init=if_initial_prefix.shape().size
        )
        if_run_stack_len = pipe(m, wb_next_run_stack_len)
        if_potential_len = pipe(m, wb_next_potential_len)
        if_decisions_len = pipe(m, wb_next_decisions_len)
        if_potential_areas = [
            pipe(m, wb_next_potential_areas[i]) for i in range(self._cuboids)
        ]

        if_req_pause = Array(self.interfaces)[if_finder].req_pause
        if_req_split = Array(self.interfaces)[if_finder].req_split

        if_prefix_done = if_prefix_bits_left == 0

        if_next_target = Signal(instruction_ref_layout(self._max_area))
        # TODO: this is the same as `if_next_inst + 1` (if we switch around the field
        # order). I think this is clearer, but switch to that if it ends up improving
        # performance.
        with m.If(if_prev_inst_ref.child_index == 3):
            m.d.comb += if_next_target.parent.eq(if_prev_inst_ref.parent + 1)
            m.d.comb += if_next_target.child_index.eq(0)
        with m.Else():
            m.d.comb += if_next_target.parent.eq(if_prev_inst_ref.parent)
            m.d.comb += if_next_target.child_index.eq(if_prev_inst_ref.child_index + 1)

        # If WB stage processed its target, we can move on to the next one, otherwise we
        # need to keep trying to process `if_prev_target`.
        if_target = data.View(
            instruction_ref_layout(self._max_area),
            Mux(if_prev_target_processed, if_next_target, if_prev_target),
        )

        if_backtrack = (if_run_stack_len != 0) & (if_target.parent == if_run_stack_len)

        if_potential_index = Mux(
            if_prev_state == State.Check, if_prev_potential_index + ~if_prev_clearing, 0
        )

        decision_sent = if_prev_sent & if_prev_prefix_done

        if_state = Signal(State)
        if_prefix = Signal.like(if_initial_prefix)
        m.d.comb += if_prefix.eq(if_initial_prefix)

        def run_transitions(allow_check: bool = True):
            """
            Rather than just writing `if_state.eq(State.Run)`, you should invoke this
            function in order to make sure that we don't end up accidentally spending an
            iteration in Run state when we should be doing something else.
            """

            with m.If(if_req_pause):
                m.d.comb += if_state.eq(State.Pause)
            with m.Elif(
                if_req_split & (if_initial_prefix.base_decision < if_decisions_len)
            ):
                m.d.comb += if_state.eq(State.Split)
                # Set `base_decision` to what the base decision of the finder we're sending will
                # be (1 past the end of its decisions), so that it gets sent out along with the
                # rest of the prefix.
                #
                # However, it might not be our new base decision, since it might be a 0: we'll
                # fix it up once we find the first 1 past our old base decision.
                m.d.comb += if_prefix.base_decision.eq(
                    if_initial_prefix.base_decision + 1
                )
            if allow_check:
                with m.Elif(
                    if_backtrack
                    & (if_run_stack_len + if_potential_len >= if_initial_prefix.area)
                ):
                    # There are enough run + potential instructions that we might have a solution,
                    # so check for that before we backtrack.
                    m.d.comb += if_state.eq(State.Check)
            with m.Else():
                # Note that this also covers the case where we backtrack immediately.
                m.d.comb += if_state.eq(State.Run)

        with m.Switch(if_prev_state):
            with m.Case(State.Clear):
                with m.If(if_prev_clear_index == shard_depth(self._max_area) - 1):
                    m.d.comb += if_state.eq(State.Receive)
                with m.Else():
                    m.d.comb += if_state.eq(State.Clear)
            with m.Case(State.Receive):
                with m.If(if_prev_received & if_prev_in.last):
                    run_transitions()
                with m.Else():
                    m.d.comb += if_state.eq(State.Receive)
            with m.Case(State.Run):
                run_transitions()
            with m.Case(State.Check):
                all_squares_filled = Cat(
                    if_run_stack_len + if_potential_areas[i] == if_initial_prefix.area
                    for i in range(self._cuboids)
                ).all()
                with m.If(~if_prev_clearing & all_squares_filled):
                    # All the squares are filled, which means we have a potential solution!
                    m.d.comb += if_state.eq(State.Solution)
                # Note: although `wb_potential_index` can only go up to `max_potential_len - 1`,
                # this can go all the way up to `max_potential_len` thanks to Amaranth inferring
                # a shape big enough to fit all possible values of `if_prev_potential_index +
                # ~if_prev_clearing`.
                with m.Elif(if_potential_index == if_potential_len):
                    # We've run all the potential instructions and not all the squares are filled,
                    # so this isn't a solution. Time to backtrack.
                    run_transitions(allow_check=False)
                with m.Else():
                    m.d.comb += if_state.eq(State.Check)
            with m.Case(State.Solution):
                with m.If(
                    (if_prev_finder_done | decision_sent)
                    & (if_prev_decision_index == if_decisions_len - 1)
                ):
                    run_transitions(allow_check=False)
                with m.Else():
                    m.d.comb += if_state.eq(State.Solution)
            with m.Case(State.Pause):
                # We transition out of `State.Pause` via. `local_reset`, rather than via. a
                # regular state transition.
                m.d.comb += if_state.eq(State.Pause)
            with m.Case(State.Split):
                # We don't actually stop once the finder is sent like you might expect: since
                # `base_decision` can't point to a 0, we have to keep going until we find a 1 to
                # set it to, or hit the end of `decisions` if there aren't any.
                with m.If(
                    (if_prev_finder_done & if_prev_read_decision)
                    | (decision_sent & (if_prev_decision_index == if_decisions_len - 1))
                ):
                    run_transitions()
                with m.Else():
                    m.d.comb += if_state.eq(State.Split)

        # We only get to Clear state via. resetting, which will reset this to 0 anyway:
        # so the only time we actually need to reset it is when exiting Check state so
        # that we don't waste time clearing addresses the potential surfaces don't have.
        if_clear_index = Mux(
            (if_prev_state == State.Check) & (if_state != State.Check),
            0,
            if_prev_clear_index + 1,
        )

        if_decision_index = Signal(range(max_decisions_len(self._max_area)))
        # Note: technically this is still incorrect, since it'd be possible to go
        # directly from one split into another if `req_split` was already asserted when
        # finishing the first one. That shouldn't happen in practice since our SoC won't
        # ask anyone to split while there's already a split happening, but we might
        # still want to fix it at some point.
        with m.If(if_state != if_prev_state):
            m.d.comb += if_decision_index.eq(0)
        with m.Else():
            m.d.comb += if_decision_index.eq(
                if_prev_decision_index
                + (decision_sent | ((if_state == State.Split) & if_prev_finder_done))
            )

        m.d.comb += potential_read.chunk.eq(if_finder)
        m.d.comb += potential_read.addr.eq(if_potential_index)

        if_run_stack_index = Signal(range(max_run_stack_len(self._max_area)))
        with m.If(if_state == State.Check):
            m.d.comb += if_run_stack_index.eq(potential_read.data.parent)
        with m.Elif(if_backtrack):
            m.d.comb += if_run_stack_index.eq(if_run_stack_len - 1)
        with m.Else():
            m.d.comb += if_run_stack_index.eq(if_target.parent)

        if_child_index = Mux(
            if_state == State.Check,
            potential_read.data.child_index,
            if_target.child_index,
        )

        if_in_fifo = Array(in_fifos)[if_finder]
        if_in_rdy = if_in_fifo.r_rdy
        if_in = data.View(core_in_layout(), if_in_fifo.r_data)

        if_task = Signal(Task)
        with m.Switch(if_state):
            with m.Case(State.Clear):
                m.d.comb += if_task.eq(Task.Clear)
            with m.Case(State.Receive):
                m.d.comb += if_task.eq(
                    # If we've received a decision of 1, we need to run the next valid instruction
                    # to fulfil it; otherwise, we want to not run the next valid instruction, but we
                    # still need to check whether it was valid so that we know whether we can move
                    # on to the next decision.
                    Mux(
                        if_prefix_done & if_in_rdy & if_in.data,
                        Task.Advance,
                        # This also serves as a no-op in the case where there isn't another bit
                        # available yet.
                        Task.Check,
                    )
                )
            with m.Case(State.Run):
                m.d.comb += if_task.eq(Mux(if_backtrack, Task.Backtrack, Task.Advance))
            with m.Case(State.Check):
                m.d.comb += if_task.eq(Task.Check)
            with m.Default():
                # In states that don't need the main pipeline, give it the Check task, since it
                # doesn't have any side effects.
                m.d.comb += if_task.eq(Task.Check)

        run_stack_read = run_stack.read_port()
        m.d.comb += run_stack_read.addr.eq(Cat(if_run_stack_index, if_finder))

        # NL

        nl_finder = (FINDERS_PER_CORE + 1 - finder_offset) % FINDERS_PER_CORE

        nl_initial_state = pipe(m, if_state)
        nl_initial_prefix_bits_left = pipe(
            m, if_prefix_bits_left, init=if_prefix.shape().size
        )
        nl_initial_decisions_len = pipe(m, if_decisions_len)
        nl_initial_task = pipe(m, if_task)
        nl_prefix = pipe(m, if_prefix)
        nl_decision_index = pipe(m, if_decision_index)
        nl_entry = run_stack_read.data
        local_reset = (
            (nl_initial_state == State.Pause)
            & (nl_initial_prefix_bits_left == 0)
            & (nl_decision_index == nl_initial_decisions_len)
        ) | (
            (nl_initial_task == Task.Backtrack)
            & (nl_entry.decision_index < nl_prefix.base_decision)
        )

        nl_state = Mux(local_reset, State.Clear, nl_initial_state)
        nl_prefix_bits_left = Mux(
            local_reset, nl_prefix.shape().size, nl_initial_prefix_bits_left
        )
        nl_run_stack_len = Mux(local_reset, 0, pipe(m, if_run_stack_len))
        nl_potential_len = Mux(local_reset, 0, pipe(m, if_potential_len))
        nl_decisions_len = Mux(local_reset, 0, nl_initial_decisions_len)
        nl_potential_areas = [
            Mux(local_reset, 0, pipe(m, if_potential_areas[i]))
            for i in range(self._cuboids)
        ]
        nl_prefix_done = nl_prefix_bits_left == 0
        nl_target = Mux(local_reset, 0, pipe(m, if_target))
        nl_potential_index = pipe(m, if_potential_index)
        nl_clear_index = Mux(local_reset, 0, pipe(m, if_clear_index))
        nl_child_index = pipe(m, if_child_index)
        nl_in_rdy = pipe(m, if_in_rdy)
        nl_in = pipe(m, if_in)
        nl_task = Mux(local_reset, Task.Clear, nl_initial_task)

        main_pipeline = MainPipeline(self._cuboids, self._max_area)
        m.submodules.main_pipeline = main_pipeline

        m.d.comb += main_pipeline.finder.eq(nl_finder)
        m.d.comb += main_pipeline.start_mapping_index.eq(nl_prefix.start_mapping_index)
        m.d.comb += main_pipeline.task.eq(nl_task)
        m.d.comb += main_pipeline.entry.eq(nl_entry)
        with m.If(nl_run_stack_len == 0):
            m.d.comb += main_pipeline.entry.instruction.pos.x.eq(0)
            m.d.comb += main_pipeline.entry.instruction.pos.y.eq(0)
            m.d.comb += main_pipeline.entry.instruction.mapping.eq(
                nl_prefix.start_mapping
            )
        m.d.comb += main_pipeline.child.eq(
            (nl_task != Task.Backtrack) & (nl_run_stack_len != 0)
        )
        m.d.comb += main_pipeline.child_index.eq(nl_child_index)
        m.d.comb += main_pipeline.clear_index.eq(nl_clear_index)
        for i in range(self._cuboids):
            wiring.connect(
                m,
                main_pipeline.neighbour_lookups[i],
                wiring.flipped(self.neighbour_lookups[i]),
            )
        for i in range(self._cuboids - 1):
            wiring.connect(
                m,
                main_pipeline.undo_lookups[i],
                wiring.flipped(self.undo_lookups[i]),
            )

        # VC

        vc_finder = (FINDERS_PER_CORE + 2 - finder_offset) % FINDERS_PER_CORE

        vc_state = pipe(m, nl_state)
        vc_prefix = pipe(m, nl_prefix)
        vc_prefix_bits_left = pipe(m, nl_prefix_bits_left, init=vc_prefix.shape().size)
        vc_run_stack_len = pipe(m, nl_run_stack_len)
        vc_potential_len = pipe(m, nl_potential_len)
        vc_decisions_len = pipe(m, nl_decisions_len)
        vc_potential_areas = [
            pipe(m, nl_potential_areas[i]) for i in range(self._cuboids)
        ]
        vc_prefix_done = pipe(m, nl_prefix_done)
        vc_target = pipe(m, nl_target)
        vc_potential_index = pipe(m, nl_potential_index)
        vc_decision_index = pipe(m, nl_decision_index)
        vc_clear_index = pipe(m, nl_clear_index)
        vc_in_rdy = pipe(m, nl_in_rdy)
        vc_in = pipe(m, nl_in)
        vc_task = pipe(m, nl_task)
        vc_entry = pipe(m, nl_entry)

        vc_instruction = main_pipeline.instruction

        # The `vc_potential_index == 0` is necessary because otherwise Check state would
        # freeze up as soon as it ran the first potential instruction and the potential
        # surfaces weren't empty anymore.
        vc_clearing = (vc_state != State.Check) | (
            (vc_potential_index == 0)
            & Cat(vc_potential_areas[i] != 0 for i in range(self._cuboids)).any()
        )

        for i in range(self._cuboids):
            read_port, _ = potential_surface_ports[i]
            m.d.comb += read_port.chunk.eq(vc_finder)
            m.d.comb += read_port.addr.eq(
                Mux(
                    vc_clearing,
                    vc_clear_index,
                    vc_instruction.mapping[i].square,
                )
            )

        m.d.comb += decisions_read.chunk.eq(vc_finder)
        m.d.comb += decisions_read.addr.eq(vc_decision_index)

        # WB

        wb_finder = FINDERS_PER_CORE - 1 - finder_offset

        m.d.sync += wb_state.eq(vc_state)
        wb_prefix = pipe(m, vc_prefix)
        wb_prefix_bits_left = pipe(m, vc_prefix_bits_left, init=wb_prefix.shape().size)
        wb_run_stack_len = pipe(m, vc_run_stack_len)
        wb_potential_len = pipe(m, vc_potential_len)
        wb_decisions_len = pipe(m, vc_decisions_len)
        wb_potential_areas = [
            pipe(m, vc_potential_areas[i]) for i in range(self._cuboids)
        ]
        m.d.sync += wb_prefix_done.eq(vc_prefix_done)
        m.d.sync += wb_target.eq(vc_target)
        m.d.sync += wb_potential_index.eq(vc_potential_index)
        m.d.sync += wb_decision_index.eq(vc_decision_index)
        m.d.sync += wb_clear_index.eq(vc_clear_index)
        wb_in_rdy = pipe(m, vc_in_rdy)
        m.d.sync += wb_in.eq(vc_in)
        wb_task = pipe(m, vc_task)
        wb_entry = pipe(m, vc_entry)
        wb_instruction = pipe(m, vc_instruction)
        m.d.sync += wb_clearing.eq(vc_clearing)

        wb_instruction_valid = main_pipeline.instruction_valid
        wb_neighbours_valid = main_pipeline.neighbours_valid

        m.d.comb += wb_read_decision.eq(decisions_read.data)

        wb_run = (
            (wb_task == Task.Advance) & wb_instruction_valid & wb_neighbours_valid.any()
        )
        # Potential instructions still need to be added in Receive state even if the
        # next decision is 0.
        wb_potential = (
            ((wb_task == Task.Advance) | ((wb_state == State.Receive) & wb_prefix_done))
            & wb_instruction_valid
            & ~wb_neighbours_valid.any()
        )

        m.d.comb += wb_received.eq(
            (wb_state == State.Receive)
            & wb_in_rdy
            # If the instruction wasn't valid, whether or not to run it wasn't a decision.
            & ~(wb_prefix_done & (~wb_instruction_valid | ~wb_neighbours_valid.any()))
        )
        for i in range(FINDERS_PER_CORE):
            m.d.comb += in_fifos[i].r_en.eq((wb_finder == i) & wb_received)

        # The `wb_prefix_done` is needed so that we don't try and interpret the meaning
        # of `wb_prefix.base_decision` while the prefix is still being shifted out.
        m.d.comb += wb_finder_done.eq(
            wb_prefix_done & (wb_decision_index >= wb_prefix.base_decision)
        )

        wb_out_fifo = Array(out_fifos)[wb_finder]
        m.d.comb += wb_sent.eq(
            (
                (wb_state == State.Solution)
                | (wb_state == State.Pause)
                | ((wb_state == State.Split) & ~wb_finder_done)
            )
            & wb_out_fifo.w_rdy
        )

        split_reached = (wb_state == State.Split) & (
            wb_decision_index == wb_prefix.base_decision - 1
        )
        wb_out = Signal(core_out_layout())
        m.d.comb += wb_out.data.eq(
            Mux(
                wb_prefix_done,
                wb_read_decision & ~split_reached,
                wb_prefix.as_value()[-1],
            )
        )
        m.d.comb += wb_out.last.eq(
            Mux(
                wb_prefix_done,
                (wb_decision_index == wb_decisions_len - 1) | split_reached,
                (wb_decisions_len == 0) & (wb_prefix_bits_left == 1),
            )
        )
        with m.Switch(wb_state):
            with m.Case(State.Solution):
                m.d.comb += wb_out.type.eq(FinderType.Solution)
            with m.Case(State.Pause):
                m.d.comb += wb_out.type.eq(FinderType.Pause)
            with m.Case(State.Split):
                m.d.comb += wb_out.type.eq(FinderType.Split)
            # It doesn't really matter what this is in other states, leave it as 0.

        for i in range(FINDERS_PER_CORE):
            m.d.comb += out_fifos[i].w_data.eq(wb_out)
            m.d.comb += out_fifos[i].w_en.eq((wb_finder == i) & wb_sent)

            m.d.comb += self.interfaces[i].stepping.eq(
                (wb_finder == i) & (wb_run | (wb_task == Task.Backtrack))
            )

        last_child = wb_target.child_index == sum(wb_entry.children) - 1
        normalised_target = Signal.like(wb_target)
        m.d.comb += normalised_target.parent.eq(wb_target.parent)
        m.d.comb += normalised_target.child_index.eq(
            Mux(last_child, 3, wb_target.child_index)
        )

        run_stack_write = run_stack.write_port()
        m.d.comb += run_stack_write.addr.eq(Cat(wb_run_stack_len, wb_finder))
        m.d.comb += run_stack_write.data.instruction.eq(wb_instruction)
        m.d.comb += run_stack_write.data.source.eq(normalised_target)
        m.d.comb += run_stack_write.data.children.eq(wb_neighbours_valid)
        m.d.comb += run_stack_write.data.potential_len.eq(wb_potential_len)
        m.d.comb += run_stack_write.data.decision_index.eq(wb_decisions_len)
        m.d.comb += run_stack_write.en.eq(wb_run)

        m.d.comb += potential_write.chunk.eq(wb_finder)
        m.d.comb += potential_write.addr.eq(wb_potential_len)
        m.d.comb += potential_write.data.eq(wb_target)
        m.d.comb += potential_write.en.eq(wb_potential)

        m.d.comb += decisions_write.chunk.eq(wb_finder)
        m.d.comb += decisions_write.addr.eq(
            Mux(wb_task == Task.Backtrack, wb_entry.decision_index, wb_decisions_len)
        )
        m.d.comb += decisions_write.data.eq(
            Mux(wb_state == State.Receive, wb_in.data, wb_task != Task.Backtrack)
        )
        m.d.comb += decisions_write.en.eq(
            (wb_received & wb_prefix_done) | wb_run | (wb_task == Task.Backtrack)
        )

        for i in range(self._cuboids):
            _, write_port = potential_surface_ports[i]
            m.d.comb += write_port.chunk.eq(wb_finder)
            m.d.comb += write_port.addr.eq(
                Mux(
                    wb_clearing,
                    wb_clear_index,
                    wb_instruction.mapping[i].square,
                )
            )
            m.d.comb += write_port.data.eq(~wb_clearing)
            m.d.comb += write_port.en.eq(
                Mux(
                    wb_clearing,
                    wb_clear_index < self._max_area,
                    wb_instruction_valid,
                )
            )

        shift_prefix = (wb_received | wb_sent) & ~wb_prefix_done
        prefix_in = Mux(wb_state == State.Receive, wb_in.data, wb_prefix.as_value()[-1])
        m.d.comb += wb_next_prefix.eq(
            Mux(shift_prefix, Cat(prefix_in, wb_prefix.as_value()[:-1]), wb_prefix)
        )
        m.d.comb += wb_next_prefix_bits_left.eq(
            Mux(
                ((wb_state == State.Receive) & ~(wb_in.last & wb_received))
                | ((wb_state == State.Solution) & ~(wb_out.last & wb_sent))
                | (wb_state == State.Pause)
                | (
                    (wb_state == State.Split)
                    & ~(
                        (wb_finder_done & wb_read_decision)
                        | (
                            (wb_finder_done | (wb_prefix_done & wb_sent))
                            & (wb_decision_index == wb_decisions_len - 1)
                        )
                    )
                ),
                wb_prefix_bits_left - shift_prefix,
                wb_prefix.shape().size,
            )
        )

        with m.If(
            (wb_task == Task.Backtrack)
            & (wb_entry.decision_index == wb_prefix.base_decision)
        ):
            m.d.comb += wb_next_prefix.base_decision.eq(wb_prefix.base_decision + 1)
        with m.Elif((wb_state == State.Split) & wb_finder_done & wb_read_decision):
            m.d.comb += wb_next_prefix.base_decision.eq(wb_decision_index)
        with m.Elif(
            (wb_state == State.Split)
            # Unlike the previous case, this can actually occur on the same cycle as
            # wb_out.last is set, so just wb_finder_done wouldn't work here.
            & (wb_finder_done | (wb_prefix_done & wb_sent))
            & (wb_decision_index == wb_decisions_len - 1)
        ):
            m.d.comb += wb_next_prefix.base_decision.eq(wb_decisions_len)

        with m.If(wb_task == Task.Backtrack):
            m.d.comb += wb_next_run_stack_len.eq(wb_run_stack_len - 1)
            m.d.comb += wb_next_potential_len.eq(wb_entry.potential_len)
        with m.Else():
            m.d.comb += wb_next_run_stack_len.eq(wb_run_stack_len + wb_run)
            m.d.comb += wb_next_potential_len.eq(wb_potential_len + wb_potential)

        with m.Switch(wb_state):
            with m.Case(State.Receive):
                m.d.comb += wb_next_decisions_len.eq(
                    wb_decisions_len + (wb_received & wb_prefix_done)
                )
            with m.Case(State.Run):
                with m.If(wb_task == Task.Backtrack):
                    m.d.comb += wb_next_decisions_len.eq(wb_entry.decision_index + 1)
                with m.Else():
                    m.d.comb += wb_next_decisions_len.eq(wb_decisions_len + wb_run)
            with m.Default():
                m.d.comb += wb_next_decisions_len.eq(wb_decisions_len)

        for i in range(self._cuboids):
            read_port, write_port = potential_surface_ports[i]
            with m.If(wb_state == State.Clear):
                m.d.comb += wb_next_potential_areas[i].eq(wb_potential_areas[i])
            with m.Elif(write_port.en & (read_port.data == 0) & (write_port.data == 1)):
                m.d.comb += wb_next_potential_areas[i].eq(wb_potential_areas[i] + 1)
            with m.Elif(write_port.en & (read_port.data == 1) & (write_port.data == 0)):
                m.d.comb += wb_next_potential_areas[i].eq(wb_potential_areas[i] - 1)
            with m.Else():
                m.d.comb += wb_next_potential_areas[i].eq(wb_potential_areas[i])

        m.d.comb += wb_target_processed.eq(
            (
                ((wb_state == State.Receive) & wb_prefix_done & wb_in_rdy)
                | (wb_state == State.Run)
            )
            & (wb_run_stack_len != 0)
        )

        # Arguably we should consider Check state here too, but this is only used for
        # computing the next target anyway so it doesn't really matter.
        #
        # We use `normalised_target` here so that we can always find the next target by
        # just adding 1 to `child_index`.
        m.d.comb += wb_inst_ref.eq(
            Mux(wb_task == Task.Backtrack, wb_entry.source, normalised_target)
        )

        for i in range(FINDERS_PER_CORE):
            state = Signal(State)
            prefix = Signal.like(if_prefix)
            prefix_bits_left = Signal.like(if_prefix_bits_left)
            with m.Switch((finder_offset + i) % FINDERS_PER_CORE):
                with m.Case(0):
                    m.d.comb += state.eq(if_state)
                    m.d.comb += prefix.eq(if_prefix)
                    m.d.comb += prefix_bits_left.eq(if_prefix_bits_left)
                with m.Case(1):
                    m.d.comb += state.eq(nl_state)
                    m.d.comb += prefix.eq(nl_prefix)
                    m.d.comb += prefix_bits_left.eq(nl_prefix_bits_left)
                with m.Case(2):
                    m.d.comb += state.eq(vc_state)
                    m.d.comb += prefix.eq(vc_prefix)
                    m.d.comb += prefix_bits_left.eq(vc_prefix_bits_left)
                with m.Case(3):
                    m.d.comb += state.eq(wb_state)
                    m.d.comb += prefix.eq(wb_prefix)
                    m.d.comb += prefix_bits_left.eq(wb_prefix_bits_left)

            m.d.comb += self.interfaces[i].wants_finder.eq(
                (state == State.Receive) & (prefix_bits_left == prefix.shape().size)
            )
            m.d.comb += self.interfaces[i].state.eq(state)
            m.d.comb += self.interfaces[i].base_decision.eq(prefix.base_decision)

        m.d.comb += self.state.eq(wb_state)

        return m
