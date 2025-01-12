from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.memory import WritePort
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from .core import Core, FinderType, State, core_in_layout, core_out_layout
from .neighbour_lookup import neighbour_lookup_layout
from .skip_checker import undo_lookup_layout
from .utils import Merge, PipeReady, PipeValid, SafeDemux
from .memory import ConfigMemory


def core_group_out_layout():
    return data.StructLayout(
        {
            "data": 1,
            "last": 1,
            "kind": 1,
        }
    )


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
                # Whether any of the cores in this group still have work left to do.
                "active": Out(1),
                # The number of new finders that have been produced via. splitting.
                "split_finders": Out(32),
                # The total number of finders that have finished running.
                "completed_finders": Out(32),
            }
            | {f"{state.lower()}_count": Out(64) for state in State.__members__}
        )

        self._cuboids = cuboids
        self._max_area = max_area
        self._n = n

    def elaborate(self, platform) -> Module:
        m = Module()

        neighbour_lookups = []
        for i, port in enumerate(self.neighbour_lookups):
            # TODO: this should be a regular Memory: using ConfigMemory requires using the
            # BRAMs in TDP mode, but since the neighbour lookups are 36 bits wide that
            # requires jumping up to a 36Kib BRAM. This doesn't actually change the resource
            # usage at all, since it halves the number of replicas we need at the same time
            # as making those replicas bigger, but if a plain memory has the same resource
            # usage then we should obviously use that rather than this hacky approach.
            neighbour_lookup = ConfigMemory(
                data=neighbour_lookup_layout(self._max_area)
            )
            m.submodules[f"neighbour_lookup_{i}"] = neighbour_lookup
            wiring.connect(m, neighbour_lookup.write_port, wiring.flipped(port))
            neighbour_lookups.append(neighbour_lookup)

        undo_lookups = []
        for i, port in enumerate(self.undo_lookups):
            undo_lookup = ConfigMemory(data=undo_lookup_layout(self._max_area))
            m.submodules[f"undo_lookup_{i}"] = undo_lookup
            wiring.connect(m, undo_lookup.write_port, wiring.flipped(port))
            undo_lookups.append(undo_lookup)

        # TODO: maybe rename to ifaces?
        cores = []
        core_sinks = []
        real_cores = []
        for i in range(self._n):
            core = Core(self._cuboids, self._max_area)
            m.submodules[f"core_{i}"] = core

            for j, interface in enumerate(core.interfaces):
                # TODO: do we still need this when there's a FIFO inside the core now?
                # Also, the throughput benefits of PipeValid/PipeReady over a FIFO are
                # completely moot when each interface can only communicate once every 4 cycles
                # now anyway. (although I think it's actually 8 cycles rn since we only consume
                # values from the FIFO in WB stage, so the new value hasn't been read in yet in
                # IF stage? Maybe we should actually do something about that.)
                pipe_valid = PipeValid(core_in_layout())
                pipe_ready = PipeReady(core_in_layout())

                m.submodules[f"core_{i}_pipe_valid_{j}"] = pipe_valid
                m.submodules[f"core_{i}_pipe_ready_{j}"] = pipe_ready

                wiring.connect(m, pipe_ready.source, interface.sink)
                wiring.connect(m, pipe_valid.source, pipe_ready.sink)
                core_sinks.append(pipe_valid.sink)

                cores.append(interface)
                m.d.sync += interface.req_pause.eq(self.req_pause)

            for lookup, port in zip(neighbour_lookups, core.neighbour_lookups):
                wiring.connect(m, lookup.read_port(), port)

            for lookup, port in zip(undo_lookups, core.undo_lookups):
                wiring.connect(m, lookup.read_port(), port)

            real_cores.append(core)

        output_mux = SafeDemux(core_out_layout(), 2)
        m.submodules.output_mux = output_mux

        # Distribute incoming finders to whichever cores want them.
        input_mux = SafeDemux(core_in_layout(), len(cores))
        m.submodules.input_mux = input_mux
        for source, sink in zip(input_mux.sources, core_sinks):
            wiring.connect(m, source, sink)

        m.d.comb += input_mux.en.eq(Cat(core.wants_finder for core in cores).any())
        for i, core in enumerate(cores):
            with m.If(core.wants_finder):
                m.d.comb += input_mux.sel.eq(i)

        # Incoming packets can come from either `self.sink` or cores splitting.
        input_merge = Merge(core_in_layout(), 2)
        m.submodules.input_merge = input_merge

        wiring.connect(m, wiring.flipped(self.sink), input_merge.sinks[0])

        m.d.comb += input_merge.sinks[1].valid.eq(output_mux.sources[1].valid)
        m.d.comb += output_mux.sources[1].ready.eq(input_merge.sinks[1].ready)
        m.d.comb += input_merge.sinks[1].p.data.eq(output_mux.sources[1].p.data)
        m.d.comb += input_merge.sinks[1].p.last.eq(output_mux.sources[1].p.last)

        wiring.connect(m, input_merge.source, input_mux.sink)

        # Most outgoing packets get merged together into `self.source`, unless they're
        # of `SPLIT` kind: then they get routed back around into `self.split`.
        output_merge = Merge(core_out_layout(), len(cores))
        m.submodules.output_merge = output_merge

        # Put a buffer in front of `output_merge` so that it's impossible for the
        # outputs of cores to affect the inputs of others.
        pipe_valid = PipeValid(core_out_layout())
        pipe_ready = PipeReady(core_out_layout())
        m.submodules.output_merge_pipe_valid = pipe_valid
        m.submodules.output_merge_pipe_ready = pipe_ready
        wiring.connect(m, output_merge.source, pipe_valid.sink)
        wiring.connect(m, pipe_valid.source, pipe_ready.sink)

        for i, core in enumerate(cores):
            wiring.connect(m, cores[i].source, output_merge.sinks[i])
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

        # Whether or not each core is currently running something.
        cores_active = Signal(len(cores))
        m.d.sync += cores_active.eq(
            Cat((core.state != State.Clear) & ~core.wants_finder for core in cores)
        )
        # Whether or not each core was running something during the previous clock cycle.
        prev_cores_active = Signal.like(cores_active)

        m.d.sync += prev_cores_active.eq(cores_active)

        m.d.comb += self.active.eq(self.sink.valid | cores_active.any())

        # The `base_decision` that a core has to have in order to be split.
        #
        # If no cores have this `base_decision`, it gets incremented until one does and
        # we can split.
        splittable_base = Signal.like(cores[0].base_decision)
        # Whether or not each core is splittable (is active and has a `base_decision` of
        # `splittable_base`).
        splittable = Signal(len(cores))
        m.d.sync += splittable.eq(
            Cat(
                # TODO: base_decision is garbage while sending, so this decision-making might be a bit off.
                cores_active[i] & (core.base_decision == splittable_base)
                for i, core in enumerate(cores)
            )
        )
        with m.If(
            Cat(
                ~prev_cores_active[i] & cores_active[i] for i in range(len(cores))
            ).any()
        ):
            # A new core's just become active and might have a `base_decision` below
            # `splittable_base`, so reset it to 1.
            m.d.sync += splittable_base.eq(1)
        with m.Elif(~splittable.any()):
            # There aren't any cores with this `base_decision`, so increment it until there
            # are.
            m.d.sync += splittable_base.eq(splittable_base + 1)

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
                    m.d.sync += splittee.eq(i)

        # We only want to split next clock cycle if:
        split_wanted = Signal()
        m.d.comb += split_wanted.eq(
            # - There's actually a splittable core.
            splittable.any()
            # - There aren't already more finders coming in from outside.
            & ~self.sink.valid
            # - There isn't another core that's already splitting (so that we don't
            #   accidentally split twice to fulfil one request for a finder).
            & ~Cat(core.state == State.Split for core in cores).any()
            # - There are actually cores that want finders.
            & Cat(core.wants_finder for core in cores).any()
        )

        for i, core in enumerate(cores):
            # Request that `splittee` split itself, assuming splitting is actually desired
            # in the first place.
            #
            # Do it synchronously so there isn't a timing path between the cores.
            m.d.sync += core.req_split.eq((splittee == i) & split_wanted)

        # Add 1 to `split_finders` every time a split finder passes through `output_merge`.
        with m.If(
            output_merge.source.valid
            & output_merge.source.ready
            & output_merge.source.p.last
            & (output_merge.source.p.type == FinderType.Split)
        ):
            m.d.sync += self.split_finders.eq(self.split_finders + 1)

        m.d.sync += self.completed_finders.eq(
            self.completed_finders
            + sum(
                # If the core was previously active and now isn't, that means it must have just
                # finished running its finder.
                (prev_cores_active[i] & ~cores_active[i] for i in range(len(cores)))
            )
        )

        for name, state in State.__members__.items():
            counter = getattr(self, f"{name.lower()}_count")
            m.d.sync += counter.eq(
                counter + sum(core.state == state for core in real_cores)
            )

        return m
