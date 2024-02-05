from litescope import LiteScopeAnalyzer
from litex.gen import *
from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import *
from migen.genlib.cdc import MultiReg
from migen.genlib.coding import Encoder, PriorityEncoder
from migen.genlib.fifo import AsyncFIFO
from utils import Collect, Flatten, Merge, SafeDemux


class Cuboid:
    def __init__(self, cuboid: str):
        self.width, self.depth, self.height = map(int, cuboid.split("x"))

    def surface_area(self) -> int:
        """Returns the surface area of this cuboid."""
        return 2 * (
            self.width * self.depth
            + self.width * self.height
            + self.depth * self.height
        )

    def num_classes(self) -> int:
        """
        Returns the number of equivalence classes of cursors on this cuboid.

        Two cursors are considered 'equivalent' if you can rotate one cursor into the
        position of the other and the cuboid looks identical to how it did before you
        rotated it.
        """
        cursors = 4 * self.surface_area()
        if self.width != self.depth != self.height:
            # Each cursor is equivalent to its rotation 180 degrees on the same face, as
            # well as both it and its rotation's versions on the opposite face.
            return cursors // 4
        elif self.width == self.depth == self.height:
            # Each cursor is equivalent to itself, all 3 90-degree rotations of itself on
            # the same face, and the versions of itself and its rotations on all the other
            # faces. 4 * 6 = 24.
            return cursors // 24
        else:
            # If the cursor is on the square face, it's equivalent to all 3 90-degree
            # rotations of itself on the same face and their equivalents on the opposite
            # face.
            #
            # Otherwise, there are 3 faces the same shape as the face it's on, so even
            # though it's only equivalent to its 180-degree rotation on the same face it's
            # then equivalent to the copies of those on the other 3 faces.
            #
            # So either way, each cursor has 8 equivalents.
            return cursors // 8

    def __str__(self) -> str:
        return f"{self.width}x{self.depth}x{self.height}"


# Values for `core.source.kind`.
SOLUTION = 0
SPLIT = 1
PAUSE = 2


class Core(LiteXModule):
    """A LiteX wrapper around `core`."""

    def __init__(self):
        self.sink = stream.Endpoint([("data", 1)])
        """A stream of finders for the core to run."""
        self.source = stream.Endpoint(
            stream.EndpointDescription([("data", 1)], [("kind", 2)])
        )
        """
        A stream of finders that the core outputs.

        Use `out_solution`, `out_split` and `out_pause` to figure out what each finder's
        purpose is.
        """

        self.req_pause = Signal()
        """
        Set this to 1 to ask the core to pause itself.

        It might not respond right away if it's in the middle of something, so it needs
        to be held as 1 until `out_pause` is asserted.
        """
        self.req_split = Signal()
        """
        Set this to 1 to ask the core to split itself.

        It might not respond right away if it's in the middle of something, so it needs
        to be held as 1 until `out_split` is asserted.
        """

        self.wants_finder = Signal()
        """
        Whether this core needs a new finder to run.

        This is asserted when the core enters RECEIVE state, and deasserted as soon
        as it starts receiving a finder.
        """

        self.state = Record(
            [
                # Keep this in sync with the definition in `state_t` in `core.sv` (it's in
                # reverse order because Migen goes from LSB to MSB, whereas SystemVerilog goes
                # from MSB to LSB).
                ("pause", 1),
                ("split", 1),
                ("stall", 1),
                ("solution", 1),
                ("check", 1),
                ("check_wait", 1),
                ("backtrack", 1),
                ("run", 1),
                ("receive", 1),
                ("clear", 1),
            ]
        )
        """
        The current state the core's in, for profiling and/or debugging purposes.
        """

        out_solution = Signal()
        out_split = Signal()
        out_pause = Signal()

        # # #

        self.specials += Instance(
            "core",
            i_clk=ClockSignal("core"),
            i_reset=ResetSignal("core"),
            i_in_data=self.sink.data,
            i_in_valid=self.sink.valid,
            o_in_ready=self.sink.ready,
            i_in_last=self.sink.last,
            o_out_data=self.source.data,
            o_out_valid=self.source.valid,
            i_out_ready=self.source.ready,
            o_out_last=self.source.last,
            o_out_solution=out_solution,
            o_out_split=out_split,
            o_out_pause=out_pause,
            i_req_pause=self.req_pause,
            i_req_split=self.req_split,
            o_wants_finder=self.wants_finder,
            o_state=self.state.raw_bits(),
        )

        self.enc = Encoder(3)
        self.comb += self.enc.i.eq(Cat(out_solution, out_split, out_pause))
        self.comb += self.source.kind.eq(self.enc.o)

        out_first = Signal(reset=1)
        self.comb += self.source.first.eq(out_first)
        self.sync.core += If(
            self.source.valid & self.source.ready,
            # The next bit is the first bit of a packet whenever the bit we've just
            # outputted was the last bit of the previous packet.
            out_first.eq(self.source.last),
        )


class CoreGroup(LiteXModule):
    """
    A container for a bunch of `Core`s which provides roughly the same interface as
    a single `Core`, and automatically handles splitting.

    The differences are that there's no `req_split`, and `kind` is just whether the
    packet's a result of pausing (since packets from splitting stay inside the
    `CoreGroup`).

    `req_pause` means to pause all the cores (although that'll only happen if it's
    held high long enough for them to actually do so).
    """

    def __init__(self, n: int):
        self.sink = stream.Endpoint([("data", 1)])
        self.source = stream.Endpoint(
            stream.EndpointDescription([("data", 1)], [("kind", 1)])
        )

        self.req_pause = Signal()

        self.active = Signal()
        """Whether any of the cores in this group still have work left to do."""

        # # #

        cores = []
        for i in range(n):
            core = Core()
            core = stream.BufferizeEndpoints(
                {"sink": stream.DIR_SINK}, pipe_valid=True, pipe_ready=True
            )(core)
            # The `Core` already knows that it's supposed to use the `core` clock domain,
            # but the `BufferizeEndpoints` doesn't.
            core = ClockDomainsRenamer("core")(core)
            cores.append(core)
            setattr(self, f"core{i}", core)

        # The layout of `Core.source``, which unlike `self.source` still include
        # splitting as a possibility.
        source_layout = cores[0].source.description

        output_mux = stream.Demultiplexer(source_layout, 2)
        self.output_mux = ClockDomainsRenamer("core")(output_mux)

        # Distribute incoming finders to whichever cores want them.
        input_mux = SafeDemux(self.sink.description, n)
        self.input_mux = ClockDomainsRenamer("core")(input_mux)
        for i in range(n):
            self.comb += getattr(self.input_mux, f"source{i}").connect(cores[i].sink)

        self.input_enc = PriorityEncoder(n)
        self.comb += self.input_enc.i.eq(Cat(core.wants_finder for core in cores))
        self.comb += self.input_mux.sel.eq(self.input_enc.o)

        # Incoming packets can come from either `self.sink` or cores splitting.
        input_merge = Merge(self.sink.description, 2)
        self.input_merge = ClockDomainsRenamer("core")(input_merge)
        self.comb += self.sink.connect(self.input_merge.sink0)
        self.comb += self.output_mux.source1.connect(
            self.input_merge.sink1, omit={"kind"}
        )
        self.comb += self.input_merge.source.connect(
            self.input_mux.sink, omit={"valid", "ready"}
        )

        # Don't mark the input to `input_mux` as valid unless the result of the encoder
        # is valid, or we're already locked into a particular target anyway.
        mux_en = ~self.input_enc.n | self.input_mux.active
        self.comb += self.input_mux.sink.valid.eq(
            self.input_merge.source.valid & mux_en
        )
        # Also don't propagate `ready` in that case, otherwise inputs end up
        # disappearing into thin air thinking they're being accepted when they aren't.
        self.comb += self.input_merge.source.ready.eq(
            self.input_mux.sink.ready & mux_en
        )

        # Most outgoing packets get merged together into `self.source`, unless they're
        # of `SPLIT` kind: then they get routed back around into `self.split`.
        output_merge = Merge(source_layout, n)
        # Put a buffer in front of `output_merge` so that it's impossible for the
        # outputs of cores to affect the inputs of others.
        output_merge = stream.BufferizeEndpoints(
            {"source": stream.DIR_SOURCE}, pipe_valid=True, pipe_ready=True
        )(output_merge)
        self.output_merge = ClockDomainsRenamer("core")(output_merge)
        for i in range(n):
            self.comb += cores[i].source.connect(getattr(self.output_merge, f"sink{i}"))
        self.comb += self.output_merge.source.connect(self.output_mux.sink)

        self.comb += self.output_mux.sel.eq(self.output_mux.sink.kind == SPLIT)
        self.comb += self.output_mux.source0.connect(self.source, omit={"kind"})
        self.comb += self.source.kind.eq(self.output_mux.source0.kind == PAUSE)

        for i, core in enumerate(cores):
            # Request that the previous core split itself whenever this core needs a new
            # finder, there aren't any more coming in from outside, and the `Merge` on the
            # core's output is actually free to receive it.
            #
            # Do it synchronously so there isn't a timing path between the cores.
            self.sync.core += cores[i - 1].req_split.eq(
                core.wants_finder & ~self.sink.valid & self.output_merge.source.ready
            )

        self.comb += self.active.eq(
            self.sink.valid | Reduce("OR", (~core.wants_finder for core in cores))
        )


class CoreManager(LiteXModule):
    """
    A wrapper around one or more `Core`s that provides a CSR-based interface to
    them.
    """

    def __init__(self, cuboids: list[Cuboid], cores: int, with_analyzer: bool = False):
        self.cores = CoreGroup(cores)

        area = cuboids[0].surface_area()

        square_bits = bits_for(area - 1)
        cursor_bits = square_bits + 2
        mapping_bits = len(cuboids) * cursor_bits
        mapping_index_bits = sum(
            bits_for(cuboid.num_classes() - 1) for cuboid in cuboids
        )
        decision_index_bits = bits_for(4 * area - 1)

        prefix_bits = mapping_bits + mapping_index_bits + decision_index_bits
        finder_bits = prefix_bits + 4 * area

        self.input = CSRStorage(
            description="An input to send to a `core`.",
            fields=[
                CSRField(
                    "finder",
                    size=finder_bits,
                    description="The finder to send to the core.",
                ),
                CSRField(
                    "len",
                    size=bits_for(finder_bits),
                    description="How many bits long `finder` is.",
                ),
            ],
        )
        self.input_submit = CSRStorage(
            description="Write to this CSR to send `input` to a `core`."
        )

        self.output = CSRStatus(
            description="An output of a `core`.",
            fields=[
                CSRField(
                    "finder",
                    size=finder_bits,
                    description="The finder that the `core` output.",
                ),
                CSRField(
                    "len",
                    size=bits_for(finder_bits),
                    description="How many bits long `finder` is.",
                ),
                CSRField(
                    "kind",
                    description="What kind of output it is.",
                    values=[
                        (
                            "0",
                            "solution",
                            "The core's outputting a potential solution it found.",
                        ),
                        (
                            "1",
                            "pause",
                            "The core's paused and is outputting its state.",
                        ),
                    ],
                ),
            ],
        )
        self.output_consume = CSRStorage(
            description="Write to this CSR to mark that you've read `output` and it can be replaced with a new one."
        )

        self.flow = CSRStatus(
            description="Flow control for `input` & `output`.",
            fields=[
                CSRField(
                    "input_ready",
                    description="Whether it's okay to submit another finder to `input`.",
                ),
                CSRField(
                    "output_valid",
                    description="Whether there's currently an output ready to be consumed.",
                ),
            ],
        )

        self.active = CSRStatus(
            description="Whether there's still any work left for the core to do."
        )

        fifo_depth = 256

        # Code for getting finders from `input` to the cores.
        input_fifo = AsyncFIFO(self.input.size, fifo_depth)
        self.input_fifo = ClockDomainsRenamer({"write": "sys", "read": "core"})(
            input_fifo
        )

        self.comb += self.input_fifo.din.eq(self.input.storage)
        self.comb += self.flow.fields.input_ready.eq(self.input_fifo.writable)
        self.comb += self.input_fifo.we.eq(self.input_submit.re)

        input_conv = Flatten(finder_bits)
        self.input_conv = ClockDomainsRenamer("core")(input_conv)
        self.comb += self.input_conv.source.connect(self.cores.sink)

        input_fifo_out = Record(
            [("finder", finder_bits), ("len", bits_for(finder_bits))]
        )
        self.comb += input_fifo_out.raw_bits().eq(self.input_fifo.dout)

        self.comb += self.input_conv.sink.data.eq(input_fifo_out.finder)
        self.comb += self.input_conv.sink.len.eq(input_fifo_out.len)
        self.comb += self.input_conv.sink.valid.eq(self.input_fifo.readable)
        self.comb += self.input_conv.sink.first.eq(1)
        self.comb += self.input_conv.sink.last.eq(1)
        self.comb += self.input_fifo.re.eq(
            self.input_conv.sink.valid & self.input_conv.sink.ready
        )

        # Code for getting finders from the cores to `output`.
        output_fifo = AsyncFIFO(self.output.size, fifo_depth)
        self.output_fifo = ClockDomainsRenamer({"write": "core", "read": "sys"})(
            output_fifo
        )

        # Ideally I'd just set `self.output.status` directly instead of doing all this,
        # but it gets overridden by `self.output.fields`.
        output_fifo_out = Record(
            [
                ("finder", finder_bits, DIR_M_TO_S),
                ("len", bits_for(finder_bits), DIR_M_TO_S),
                ("kind", 1, DIR_M_TO_S),
            ]
        )
        self.comb += output_fifo_out.raw_bits().eq(self.output_fifo.dout)
        self.comb += output_fifo_out.connect(self.output.fields)

        self.comb += self.flow.fields.output_valid.eq(self.output_fifo.readable)
        self.comb += self.output_fifo.re.eq(self.output_consume.re)

        self.output_conv = ClockDomainsRenamer("core")(
            Collect(finder_bits, [("kind", 1)])
        )
        self.comb += self.cores.source.connect(self.output_conv.sink)

        output_fifo_in = Record(
            [("finder", finder_bits), ("len", bits_for(finder_bits)), ("kind", 1)]
        )
        self.comb += self.output_fifo.din.eq(output_fifo_in.raw_bits())

        self.comb += output_fifo_in.finder.eq(self.output_conv.source.data)
        self.comb += output_fifo_in.len.eq(self.output_conv.source.len)
        self.comb += output_fifo_in.kind.eq(self.output_conv.source.kind)

        self.comb += self.output_conv.source.ready.eq(self.output_fifo.writable)
        self.comb += self.output_fifo.we.eq(
            self.output_conv.source.valid & self.output_conv.source.ready
        )

        self.specials += MultiReg(self.cores.active, self.active.status, "sys")

        # Analyzer ---------------------------------------------------------------------------------
        if with_analyzer:
            analyzer_signals = [
                self.cores.sink,
                self.cores.source,
                self.cores.req_pause,
                self.cores.active,
                self.cores.input_merge.source,
                self.cores.output_merge.source,
                self.cores.output_mux.source1,
                self.input_conv.sink,
                self.output_conv.source,
            ]
            for i in range(cores):
                core = getattr(self.cores, f"core{i}")
                analyzer_signals.append(core.sink)
                analyzer_signals.append(core.source)
                analyzer_signals.append(core.req_split)
                analyzer_signals.append(core.req_pause)
                analyzer_signals.append(core.wants_finder)
                analyzer_signals.append(core.state)
            self.analyzer = LiteScopeAnalyzer(
                analyzer_signals,
                depth=512,
                clock_domain="core",
                csr_csv="core_mgr_analyzer.csv",
            )
