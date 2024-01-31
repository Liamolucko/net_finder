from migen.genlib.fifo import AsyncFIFO
from migen.genlib.cdc import MultiReg

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import *

from litescope import LiteScopeAnalyzer


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


class Core(Module):
    """A LiteX wrapper around `core`."""

    def __init__(self):
        self.sink = stream.Endpoint([("data", 1)])
        """A stream of finders for the core to run."""
        self.source = stream.Endpoint([("data", 1)])
        """
        A stream of finders that the core outputs.

        Use `out_solution`, `out_split` and `out_pause` to figure out what each finder's
        purpose is.
        """

        self.out_solution = Signal()
        """Whether the finder being emitted is a solution."""
        self.out_split = Signal()
        """Whether the finder being emitted is a response to `req_split`."""
        self.out_pause = Signal()
        """Whether the finder being emitted is a response to `req_pause`."""

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
            o_out_solution=self.out_solution,
            o_out_split=self.out_split,
            o_out_pause=self.out_pause,
            i_req_pause=self.req_pause,
            i_req_split=self.req_split,
            o_state=self.state.raw_bits(),
        )

        out_first = Signal(reset=1)
        self.comb += self.source.first.eq(out_first)
        self.sync.core += If(
            self.source.valid & self.source.ready,
            # The next bit is the first bit of a packet whenever the bit we've just
            # outputted was the last bit of the previous packet.
            out_first.eq(self.source.last),
        )


class CoreManager(LiteXModule):
    """
    A wrapper around one or more `Core`s that provides a CSR-based interface to
    them.
    """

    def __init__(self, cuboids: list[Cuboid], with_analyzer: bool = False):
        self.core = Core()

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

        # Code for getting finders from `input` to the core.
        input_fifo = AsyncFIFO(self.input.size, fifo_depth)
        self.input_fifo = ClockDomainsRenamer({"write": "sys", "read": "core"})(
            input_fifo
        )

        self.comb += self.input_fifo.din.eq(self.input.storage)
        self.comb += self.flow.fields.input_ready.eq(self.input_fifo.writable)
        self.comb += self.input_fifo.we.eq(self.input_submit.re)

        input_conv = Flatten(finder_bits)
        self.input_conv = ClockDomainsRenamer("core")(input_conv)
        self.comb += self.input_conv.source.connect(self.core.sink)

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

        # Code for getting finders from the core to `output`.
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

        self.output_conv = ClockDomainsRenamer("core")(Collect(finder_bits))
        self.comb += self.core.source.connect(
            self.output_conv.sink, omit={"valid", "ready"}
        )
        # For now, splitting is impossible, but when it becomes possible we'll want it
        # to be routed to another core, not the output FIFO.
        self.comb += self.output_conv.sink.valid.eq(
            self.core.source.valid & ~self.core.out_split
        )
        self.comb += If(
            ~self.core.out_split, self.core.source.ready.eq(self.output_conv.sink.ready)
        )

        output_fifo_in = Record(
            [("finder", finder_bits), ("len", bits_for(finder_bits)), ("kind", 1)]
        )
        self.comb += self.output_fifo.din.eq(output_fifo_in.raw_bits())

        self.comb += output_fifo_in.finder.eq(self.output_conv.source.data)
        self.comb += output_fifo_in.len.eq(self.output_conv.source.len)
        # Set `output_fifo_in.kind` to whatever kind of finder the core was outputting
        # the last time it output a bit before `self.output_conv`'s output became valid.
        #
        # `Collect` first asserts `valid` on the clock cycle after it receives its last
        # bit, meaning there's no risk of this becoming outdated.
        self.sync.core += If(
            self.core.source.valid
            & self.core.source.ready
            & ~self.output_conv.source.valid,
            output_fifo_in.kind.eq(self.core.out_pause),
        )

        self.comb += self.output_conv.source.ready.eq(self.output_fifo.writable)
        self.comb += self.output_fifo.we.eq(
            self.output_conv.source.valid & self.output_conv.source.ready
        )

        active = self.input_fifo.readable | ~self.core.sink.ready
        self.specials += MultiReg(active, self.active.status, "sys")

        # Analyzer ---------------------------------------------------------------------------------
        if with_analyzer:
            analyzer_signals = [
                self.core.sink,
                self.core.source,
                self.core.out_solution,
                self.core.out_split,
                self.core.out_pause,
                self.core.req_pause,
                self.core.req_split,
                self.core.state,
            ]
            self.analyzer = LiteScopeAnalyzer(
                analyzer_signals,
                depth=512,
                clock_domain="core",
                csr_csv="core_mgr_analyzer.csv",
            )
