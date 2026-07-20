from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.memory import Memory, WritePort
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from .core import Core, FinderType, State, core_in_layout, core_out_layout
from .main_pipeline import FINDERS_PER_CORE, max_decisions_len
from .memory import ConfigMemory
from .neighbour_lookup import neighbour_lookup_layout
from .skip_checker import undo_lookup_layout
from .utils import (
    MaskedMerge,
    Merge,
    PipeReady,
    PipeValid,
    SafeDemux,
    StreamSync,
    pipen,
    tree_sum,
)

# If we have any FPGA-spanning wires in the `core` clock domain (currently only
# the case for measuring how many cycles were spent in each state), how many
# buffers we should assert along those wires.
#
# We can increase this over time as our target clock speed gets higher.
CORE_BUFS = 2
# If we have any FPGA-spanning wires in the `sys` clock domain, how many buffers
# we should assert along those wires.
SYS_BUFS = 1


def core_group_out_layout():
    return data.StructLayout(
        {
            "data": 1,
            "last": 1,
            "kind": 1,
        }
    )


# TODO: splitting is looking really weird when running on the board, should probably investigate that
# (this was written ages ago, idk if it's still true)
class CoreGroup(wiring.Component):
    """
    A container for a bunch of `Core`s which provides roughly the same interface as
    a single `Core`, and automatically handles splitting.

    The differences are that there's no `req_split`, and `kind` is just whether the
    packet's a result of pausing (since packets from splitting stay inside the
    `CoreGroup`).

    `req_pause` means to pause all the cores (although that'll only happen if it's
    held high long enough for them to actually do so).
    """

    def __init__(self, cuboids: int, max_area: int, n: int):
        nl_layout = neighbour_lookup_layout(max_area)
        ul_layout = undo_lookup_layout(max_area)

        super().__init__(
            {
                "sink": In(stream.Signature(core_in_layout())),
                "source": Out(stream.Signature(core_group_out_layout())),
                "neighbour_lookups": Out(
                    WritePort.Signature(
                        addr_width=ceil_log2(nl_layout.depth),
                        shape=nl_layout.shape,
                    )
                ).array(cuboids),
                "undo_lookups": Out(
                    WritePort.Signature(
                        addr_width=ceil_log2(ul_layout.depth),
                        shape=ul_layout.shape,
                    )
                ).array(cuboids - 1),
                "req_pause": In(1),
                # Whether the FIFO has room for at least 2 more finders.
                "fifo_has_room": In(1),
                # Whether any of the cores in this group still have work left to do.
                "active": Out(1),
                # The number of new finders that have been produced via. splitting.
                "split_finders": Out(32),
                # The total number of finders that have finished running.
                "completed_finders": Out(32),
                # litescope debugging
                "output_merge_sel": Out(range(n * FINDERS_PER_CORE)),
                "output_merge_active": Out(1),
                "output_merge_source_valid": Out(1),
                "output_merge_source_ready": Out(1),
                "output_merge_source_payload": Out(core_out_layout()),
                "output_mux_active": Out(1),
                "output_mux_latched_sel": Out(1),
                "loopback_valid": Out(1),
                "loopback_ready": Out(1),
                "loopback_payload": Out(core_in_layout()),
                "input_merge_sel": Out(1),
                "input_merge_active": Out(1),
                "input_mux_active": Out(1),
                "input_mux_latched_sel": Out(range(n * FINDERS_PER_CORE)),
                "input_mux_sink_valid": Out(1),
                "input_mux_sink_ready": Out(1),
                "input_mux_sink_payload": Out(core_in_layout()),
                "splittable_base": Out(range(max_decisions_len(max_area) + 1)),
                "splittee": Out(range(n * FINDERS_PER_CORE)),
                "split_wanted": Out(1),
                "splittee_sink_valid": Out(1),
                "splittee_sink_ready": Out(1),
                "splittee_sink_payload": Out(core_in_layout()),
                "splittee_source_valid": Out(1),
                "splittee_source_ready": Out(1),
                "splittee_source_payload": Out(core_out_layout()),
                "splittee_req_pause": Out(1),
                "splittee_req_split": Out(1),
                "splittee_wants_finder": Out(1),
                "splittee_state": Out(State),
                "splittee_base_decision": Out(range(max_decisions_len(max_area) + 1)),
                "sender_sink_valid": Out(1),
                "sender_sink_ready": Out(1),
                "sender_sink_payload": Out(core_in_layout()),
                "sender_source_valid": Out(1),
                "sender_source_ready": Out(1),
                "sender_source_payload": Out(core_out_layout()),
                "sender_req_pause": Out(1),
                "sender_req_split": Out(1),
                "sender_wants_finder": Out(1),
                "sender_state": Out(State),
                "sender_base_decision": Out(range(max_decisions_len(max_area) + 1)),
                "receiver_sink_valid": Out(1),
                "receiver_sink_ready": Out(1),
                "receiver_sink_payload": Out(core_in_layout()),
                "receiver_source_valid": Out(1),
                "receiver_source_ready": Out(1),
                "receiver_source_payload": Out(core_out_layout()),
                "receiver_req_pause": Out(1),
                "receiver_req_split": Out(1),
                "receiver_wants_finder": Out(1),
                "receiver_state": Out(State),
                "receiver_base_decision": Out(range(max_decisions_len(max_area) + 1)),
            }
            | {f"{state.lower()}_count": Out(64) for state in State.__members__}
        )

        self._cuboids = cuboids
        self._max_area = max_area
        self._n = n

    def elaborate(self, platform) -> Module:
        m = Module()

        neighbour_lookup_read_ports = []
        for i, port in enumerate(self.neighbour_lookups):
            neighbour_lookup_read_ports.append([])
            # If a memory has more than 16 ports, Vivado gives up on trying to implement it
            # properly regardless of what kinds of ports they are. Replicate it manually.
            for j in range(self._n):
                neighbour_lookup = Memory(data=neighbour_lookup_layout(self._max_area))
                m.submodules[f"neighbour_lookup_{i}_replica_{j}"] = neighbour_lookup
                neighbour_lookup_read_ports[i].append(
                    neighbour_lookup.read_port(domain="core")
                )

                write_port = neighbour_lookup.write_port(domain="sys")
                m.d.comb += write_port.en.eq(pipen(m, port.en, SYS_BUFS, domain="sys"))
                m.d.comb += write_port.addr.eq(
                    pipen(m, port.addr, SYS_BUFS, domain="sys")
                )
                m.d.comb += write_port.data.eq(
                    pipen(m, port.data, SYS_BUFS, domain="sys")
                )

        undo_lookups = []
        for i, port in enumerate(self.undo_lookups):
            undo_lookup = DomainRenamer("core")(
                ConfigMemory(data=undo_lookup_layout(self._max_area))
            )
            m.submodules[f"undo_lookup_{i}"] = undo_lookup
            # Since we're going from a slow clock domain to a fast clock domain, the worst
            # that could happen here is that we issue the same write for multiple cycles in
            # a row, which isn't a problem.
            m.d.core += undo_lookup.write_port.en.eq(
                pipen(m, port.en, SYS_BUFS, domain="sys")
            )
            m.d.core += undo_lookup.write_port.addr.eq(
                pipen(m, port.addr, SYS_BUFS, domain="sys")
            )
            m.d.core += undo_lookup.write_port.data.eq(
                pipen(m, port.data, SYS_BUFS, domain="sys")
            )
            undo_lookups.append(undo_lookup)

        # TODO: maybe rename to ifaces?
        cores = []
        core_sinks = []
        core_sources = []
        real_cores = []
        for i in range(self._n):
            core = DomainRenamer("core")(Core(self._cuboids, self._max_area))
            m.submodules[f"core_{i}"] = core

            for j, interface in enumerate(core.interfaces):
                in_pipe_valid = DomainRenamer("sys")(PipeValid(core_in_layout()))
                in_sync = StreamSync(core_in_layout(), "sys", "core")
                out_sync = StreamSync(core_out_layout(), "core", "sys")
                out_pipe_ready = DomainRenamer("sys")(PipeReady(core_out_layout()))

                m.submodules[f"core_{i}_in_pipe_valid_{j}"] = in_pipe_valid
                m.submodules[f"core_{i}_in_sync_{j}"] = in_sync
                m.submodules[f"core_{i}_out_sync_{j}"] = out_sync
                m.submodules[f"core_{i}_out_pipe_ready_{j}"] = out_pipe_ready

                wiring.connect(m, in_pipe_valid.source, in_sync.sink)
                wiring.connect(m, in_sync.source, interface.sink)
                wiring.connect(m, interface.source, out_sync.sink)
                wiring.connect(m, out_sync.source, out_pipe_ready.sink)
                core_sinks.append(in_pipe_valid.sink)
                core_sources.append(out_pipe_ready.source)

                cores.append(interface)
                m.d.core += interface.req_pause.eq(self.req_pause)

            for j, port in enumerate(core.neighbour_lookups):
                wiring.connect(m, neighbour_lookup_read_ports[j][i], port)

            for lookup, port in zip(undo_lookups, core.undo_lookups):
                wiring.connect(m, lookup.read_port(), port)

            real_cores.append(core)

        wants_finders = Signal(len(cores))
        core_states = Signal(data.ArrayLayout(State, len(cores)))
        core_base_decisions = Signal(
            data.ArrayLayout(cores[0].base_decision.width, len(cores))
        )
        for i, core in enumerate(cores):
            # The first buffer is to synchronise into `sys` in the first place, and the rest
            # are to cross the FPGA.
            m.d.comb += wants_finders[i].eq(
                pipen(m, core.wants_finder, SYS_BUFS + 1, domain="sys")
            )
            m.d.comb += core_states[i].eq(
                pipen(m, core.state, SYS_BUFS + 1, domain="sys")
            )
            m.d.comb += core_base_decisions[i].eq(
                pipen(m, core.base_decision, SYS_BUFS + 1, domain="sys")
            )

        ### Split handling ###

        # Whether or not each core is currently running something.
        cores_active = Signal(len(cores))
        for i in range(len(cores)):
            m.d.comb += cores_active[i].eq(
                (core_states[i] != State.Clear) & ~wants_finders[i]
            )

        # Whether or not each core was running something during the previous clock cycle.
        prev_cores_active = Signal.like(cores_active)
        m.d.sys += prev_cores_active.eq(cores_active)

        m.d.comb += self.active.eq(self.sink.valid | cores_active.any())

        # The `base_decision` that a core has to have in order to be split.
        #
        # If no cores have this `base_decision`, it gets incremented until one does and
        # we can split.
        splittable_base = Signal.like(cores[0].base_decision)
        # Whether or not each core is splittable (is active and has a `base_decision` of
        # `splittable_base`).
        splittable = Signal(len(cores))
        m.d.comb += splittable.eq(
            Cat(
                cores_active[i] & (core_base_decisions[i] == splittable_base)
                | (core_states[i] == State.Split)
                for i in range(len(cores))
            )
        )
        with m.If(
            Cat(
                ~prev_cores_active[i] & cores_active[i] for i in range(len(cores))
            ).any()
        ):
            # A new core's just become active and might have a `base_decision` below
            # `splittable_base`, so reset it to 1.
            m.d.sys += splittable_base.eq(1)
        with m.Elif(~splittable.any()):
            # There aren't any cores with this `base_decision`, so increment it until there
            # are.
            m.d.sys += splittable_base.eq(splittable_base + 1)

        # The index of the core we're going to ask to split next (assuming we don't find
        # a better candidate in the meantime).
        splittee = Signal(range(len(cores)))
        # Once we pick a `splittee`, lock it in until it's no longer splittable.
        #
        # This prevents `req_split` being moved to a different core on the same clock
        # edge that the first core accepts it, causing both of them to split and the
        # second one's split finder to have nowhere to go.
        with m.If(~Array(splittable)[splittee]):
            for i, is_splittable in enumerate(splittable):
                with m.If(is_splittable):
                    m.d.sys += splittee.eq(i)

        # We only want to split next clock cycle if:
        split_wanted = Signal()
        m.d.sys += split_wanted.eq(
            # - There's actually a splittable core.
            splittable.any()
            # - There aren't already more finders coming in from outside.
            & ~self.sink.valid
            # - We aren't in the middle of pausing.
            & ~self.req_pause
            # - There isn't another core that's already splitting (so that we don't
            #   accidentally split twice to fulfil one request for a finder).
            & ~Cat(core_states[i] == State.Split for i in range(len(cores))).any()
            # - There are actually cores that want finders.
            & Cat(wants_finders[i] for i in range(len(cores))).any()
        )

        for i, core in enumerate(cores):
            # Request that `splittee` split itself, assuming splitting is actually desired
            # in the first place.
            m.d.core += core.req_split.eq(
                pipen(m, (splittee == i) & split_wanted, SYS_BUFS, domain="sys")
            )

        ### Wiring up streams ###

        output_mux = DomainRenamer("sys")(SafeDemux(core_out_layout(), 2))
        m.submodules.output_mux = output_mux

        # Distribute incoming finders to whichever cores want them.
        input_mux = DomainRenamer("sys")(SafeDemux(core_in_layout(), len(cores)))
        m.submodules.input_mux = input_mux
        for source, sink in zip(input_mux.sources, core_sinks):
            wiring.connect(m, source, sink)

        for i in range(len(cores)):
            with m.If(wants_finders[i]):
                m.d.comb += input_mux.sel.eq(i)
        m.d.comb += input_mux.en.eq(wants_finders.any())

        # Incoming packets can come from either `self.sink` or cores splitting.
        input_merge = DomainRenamer("sys")(Merge(core_in_layout(), 2))
        m.submodules.input_merge = input_merge

        wiring.connect(m, wiring.flipped(self.sink), input_merge.sinks[0])

        m.d.comb += input_merge.sinks[1].valid.eq(output_mux.sources[1].valid)
        m.d.comb += output_mux.sources[1].ready.eq(input_merge.sinks[1].ready)
        m.d.comb += input_merge.sinks[1].p.data.eq(output_mux.sources[1].p.data)
        m.d.comb += input_merge.sinks[1].p.last.eq(output_mux.sources[1].p.last)

        wiring.connect(m, input_merge.source, input_mux.sink)

        # Most outgoing packets get merged together into `self.source`, unless they're
        # of `SPLIT` kind: then they get routed back around into `self.split`.
        output_merge = DomainRenamer("sys")(MaskedMerge(core_out_layout(), len(cores)))
        m.submodules.output_merge = output_merge

        # Put a buffer in front of `output_merge` so that it's impossible for the
        # outputs of cores to affect the inputs of others.
        pipe_valid = DomainRenamer("sys")(PipeValid(core_out_layout()))
        pipe_ready = DomainRenamer("sys")(PipeReady(core_out_layout()))
        m.submodules.output_merge_pipe_valid = pipe_valid
        m.submodules.output_merge_pipe_ready = pipe_ready
        wiring.connect(m, output_merge.source, pipe_valid.sink)
        wiring.connect(m, pipe_valid.source, pipe_ready.sink)

        for i in range(len(cores)):
            wiring.connect(m, core_sources[i], output_merge.sinks[i])
            m.d.comb += output_merge.mask[i].eq(
                (splittee == i) | (~split_wanted & self.fifo_has_room)
            )
        wiring.connect(m, pipe_ready.source, output_mux.sink)

        m.d.comb += output_mux.sel.eq(output_mux.sink.p.type == FinderType.Split)
        # Which output something gets sent to is non-negotiable, we aren't going to
        # change our mind.
        m.d.comb += output_mux.en.eq(1)
        m.d.comb += self.source.valid.eq(output_mux.sources[0].valid)
        m.d.comb += output_mux.sources[0].ready.eq(self.source.ready)
        m.d.comb += self.source.p.data.eq(output_mux.sources[0].p.data)
        m.d.comb += self.source.p.last.eq(output_mux.sources[0].p.last)
        m.d.comb += self.source.p.kind.eq(
            output_mux.sources[0].p.type == FinderType.Pause
        )

        ### Stats collection ###

        # Add 1 to `split_finders` every time a split finder passes through `output_merge`.
        with m.If(
            output_merge.source.valid
            & output_merge.source.ready
            & output_merge.source.p.last
            & (output_merge.source.p.type == FinderType.Split)
        ):
            m.d.sys += self.split_finders.eq(self.split_finders + 1)

        m.d.sys += self.completed_finders.eq(
            self.completed_finders
            + tree_sum(
                # If the core was previously active and now isn't, that means it must have just
                # finished running its finder.
                [prev_cores_active[i] & ~cores_active[i] for i in range(len(cores))]
            )
        )

        real_states = Signal(data.ArrayLayout(State, len(real_cores)))
        for i, core in enumerate(real_cores):
            m.d.comb += real_states[i].eq(
                pipen(m, core.state, CORE_BUFS, domain="core")
            )
        for name, state in State.__members__.items():
            counter = getattr(self, f"{name.lower()}_count")
            core_counter = Signal.like(counter)
            state_matches = Signal(len(real_cores))
            num_matches = Signal(range(len(real_cores) + 1))
            for i in range(len(real_cores)):
                m.d.core += state_matches[i].eq(real_states[i] == state)
            m.d.core += num_matches.eq(
                tree_sum([real_states[i] == state for i in range(len(real_cores))])
            )
            m.d.core += core_counter.eq(core_counter + num_matches)
            m.d.sys += counter.eq(core_counter)

        m.d.comb += self.output_merge_sel.eq(output_merge.sel)
        m.d.comb += self.output_merge_active.eq(output_merge.active)
        m.d.comb += self.output_merge_source_valid.eq(output_merge.source.valid)
        m.d.comb += self.output_merge_source_ready.eq(output_merge.source.ready)
        m.d.comb += self.output_merge_source_payload.eq(output_merge.source.payload)
        m.d.comb += self.output_mux_active.eq(output_mux.active)
        m.d.comb += self.output_mux_latched_sel.eq(output_mux.latched_sel)
        m.d.comb += self.loopback_valid.eq(input_merge.sinks[1].valid)
        m.d.comb += self.loopback_ready.eq(input_merge.sinks[1].ready)
        m.d.comb += self.loopback_payload.eq(input_merge.sinks[1].payload)
        m.d.comb += self.input_merge_sel.eq(input_merge.sel)
        m.d.comb += self.input_merge_active.eq(input_merge.active)
        m.d.comb += self.input_mux_active.eq(input_mux.active)
        m.d.comb += self.input_mux_latched_sel.eq(input_mux.latched_sel)
        m.d.comb += self.input_mux_sink_valid.eq(input_mux.sink.valid)
        m.d.comb += self.input_mux_sink_ready.eq(input_mux.sink.ready)
        m.d.comb += self.input_mux_sink_payload.eq(input_mux.sink.payload)
        m.d.comb += self.splittable_base.eq(splittable_base)
        m.d.comb += self.splittee.eq(splittee)
        m.d.comb += self.split_wanted.eq(split_wanted)

        m.d.comb += self.splittee_sink_valid.eq(Array(core_sinks)[splittee].valid)
        m.d.comb += self.splittee_sink_ready.eq(Array(core_sinks)[splittee].ready)
        m.d.comb += self.splittee_sink_payload.eq(Array(core_sinks)[splittee].payload)
        m.d.comb += self.splittee_source_valid.eq(Array(core_sources)[splittee].valid)
        m.d.comb += self.splittee_source_ready.eq(Array(core_sources)[splittee].ready)
        m.d.comb += self.splittee_source_payload.eq(
            Array(core_sources)[splittee].payload
        )
        m.d.comb += self.splittee_req_split.eq(split_wanted)
        m.d.comb += self.splittee_wants_finder.eq(Array(wants_finders)[splittee])
        m.d.comb += self.splittee_state.eq(Array(core_states)[splittee])
        m.d.comb += self.splittee_base_decision.eq(Array(core_base_decisions)[splittee])
        m.d.comb += self.sender_sink_valid.eq(Array(core_sinks)[output_merge.sel].valid)
        m.d.comb += self.sender_sink_ready.eq(Array(core_sinks)[output_merge.sel].ready)
        m.d.comb += self.sender_sink_payload.eq(
            Array(core_sinks)[output_merge.sel].payload
        )
        m.d.comb += self.sender_source_valid.eq(
            Array(core_sources)[output_merge.sel].valid
        )
        m.d.comb += self.sender_source_ready.eq(
            Array(core_sources)[output_merge.sel].ready
        )
        m.d.comb += self.sender_source_payload.eq(
            Array(core_sources)[output_merge.sel].payload
        )
        m.d.comb += self.sender_req_split.eq(
            (splittee == output_merge.sel) & split_wanted
        )
        m.d.comb += self.sender_wants_finder.eq(Array(wants_finders)[output_merge.sel])
        m.d.comb += self.sender_state.eq(Array(core_states)[output_merge.sel])
        m.d.comb += self.sender_base_decision.eq(
            Array(core_base_decisions)[output_merge.sel]
        )
        m.d.comb += self.receiver_sink_valid.eq(
            Array(core_sinks)[input_mux.latched_sel].valid
        )
        m.d.comb += self.receiver_sink_ready.eq(
            Array(core_sinks)[input_mux.latched_sel].ready
        )
        m.d.comb += self.receiver_sink_payload.eq(
            Array(core_sinks)[input_mux.latched_sel].payload
        )
        m.d.comb += self.receiver_source_valid.eq(
            Array(core_sources)[input_mux.latched_sel].valid
        )
        m.d.comb += self.receiver_source_ready.eq(
            Array(core_sources)[input_mux.latched_sel].ready
        )
        m.d.comb += self.receiver_source_payload.eq(
            Array(core_sources)[input_mux.latched_sel].payload
        )
        m.d.comb += self.receiver_req_split.eq(
            (splittee == input_mux.latched_sel) & split_wanted
        )
        m.d.comb += self.receiver_wants_finder.eq(
            Array(wants_finders)[input_mux.latched_sel]
        )
        m.d.comb += self.receiver_state.eq(Array(core_states)[input_mux.latched_sel])
        m.d.comb += self.receiver_base_decision.eq(
            Array(core_base_decisions)[input_mux.latched_sel]
        )

        return m
