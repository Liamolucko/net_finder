from os import path

from amaranth.back import verilog
from litescope import LiteScopeAnalyzer
from litex.gen import *
from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import *
from migen.genlib.cdc import BusSynchronizer, MultiReg, PulseSynchronizer
from migen.genlib.fifo import AsyncFIFO

from ..core import group
from ..core.core import State
from ..core.main_pipeline import max_decisions_len
from ..core.neighbour_lookup import neighbour_lookup_layout
from ..core.skip_checker import undo_lookup_layout
from .utils import Collect, Flatten

# Keep this in sync with the definition in `core/core.py`.
CORE_IN_LAYOUT = [
    ("data", 1),
    ("last", 1),
]

# Keep this in sync with the definition in `core/group.py`.
CORE_GROUP_OUT_LAYOUT = [
    ("data", 1),
    ("last", 1),
    ("kind", 1),
]


def write_port_layout(width: int, depth: int):
    return [
        ("en", 1, DIR_M_TO_S),
        ("addr", bits_for(depth - 1), DIR_M_TO_S),
        ("data", width, DIR_M_TO_S),
    ]


def neighbour_lookup_port_layout(max_area: int):
    nl_layout = neighbour_lookup_layout(max_area)
    return write_port_layout(nl_layout.shape.size, nl_layout.depth)


def undo_lookup_port_layout(max_area: int):
    ul_layout = undo_lookup_layout(max_area)
    return write_port_layout(ul_layout.shape.size, ul_layout.depth)


class CoreGroup(LiteXModule):
    """A LiteX wrapper around the Amaranth implementation of CoreGroup."""

    def __init__(self, platform, cuboids: int, max_area: int, n: int):
        self.platform = platform
        self.cuboids = cuboids
        self.max_area = max_area
        self.n = n

        self.sink = stream.Endpoint([("data", 1)])
        self.source = stream.Endpoint(
            stream.EndpointDescription([("data", 1)], [("kind", 1)])
        )

        self.neighbour_lookups = [
            Record(neighbour_lookup_port_layout(max_area)) for _ in range(cuboids)
        ]
        self.undo_lookups = [
            Record(undo_lookup_port_layout(max_area)) for _ in range(cuboids - 1)
        ]

        self.req_pause = Signal()

        self.active = Signal()
        """Whether any of the cores in this group still have work left to do."""

        self.split_finders = Signal(32)
        """The number of new finders that have been produced via. splitting."""
        self.completed_finders = Signal(32)
        """The total number of finders that have finished running."""

        # Create performance counters of how many clock cycles have been spent in each
        # state.
        for state in State.__members__:
            setattr(self, f"{state.lower()}_count", Signal(64))

        # # #

        raw_sink_payload = Record(CORE_IN_LAYOUT)
        raw_source_payload = Record(CORE_GROUP_OUT_LAYOUT)

        self.specials += Instance(
            "core_group",
            i_clk=ClockSignal("core"),
            i_rst=ResetSignal("core"),
            i_sink__payload=raw_sink_payload.raw_bits(),
            i_sink__valid=self.sink.valid,
            o_sink__ready=self.sink.ready,
            o_source__payload=raw_source_payload.raw_bits(),
            o_source__valid=self.source.valid,
            i_source__ready=self.source.ready,
            **{
                f"i_neighbour_lookups__{i}__{field}": getattr(
                    self.neighbour_lookups[i], field
                )
                for field in ["en", "addr", "data"]
                for i in range(cuboids)
            },
            **{
                f"i_undo_lookups__{i}__{field}": getattr(self.undo_lookups[i], field)
                for field in ["en", "addr", "data"]
                for i in range(cuboids - 1)
            },
            i_req_pause=self.req_pause,
            o_active=self.active,
            o_split_finders=self.split_finders,
            o_completed_finders=self.completed_finders,
            **{
                f"o_{state.lower()}_count": getattr(self, f"{state.lower()}_count")
                for state in State.__members__
            },
        )

        self.comb += raw_sink_payload.data.eq(self.sink.data)
        self.comb += raw_sink_payload.last.eq(self.sink.last)

        self.comb += self.source.data.eq(raw_source_payload.data)
        self.comb += self.source.last.eq(raw_source_payload.last)
        self.comb += self.source.kind.eq(raw_source_payload.kind)

        out_first = Signal(reset=1)
        self.comb += self.source.first.eq(out_first)
        self.sync.core += If(
            self.source.valid & self.source.ready,
            # The next bit is the first bit of a packet whenever the bit we've just
            # outputted was the last bit of the previous packet.
            out_first.eq(self.source.last),
        )

    def do_finalize(self):
        verilog_path = path.join(self.platform.output_dir, "gateware", "core_group.v")
        with open(verilog_path, "w") as f:
            f.write(
                verilog.convert(
                    group.CoreGroup(self.cuboids, self.max_area, self.n),
                    name="core_group",
                )
            )
        self.platform.add_source(verilog_path)


class LookupManager(LiteXModule):
    """
    A CSR-based interface for writing to the neighbour/undo lookups.

    This is a terrible way of doing this: hooking them up to the Wishbone bus
    directly would make way more sense, but getting `sel` working is a pain for
    something which isn't at all performance-critical.
    """

    def __init__(self, ports):
        self.addr = CSRStorage(
            len(ports[0].addr),
            description="The address to write to in the chosen lookup table.",
        )
        self.data = CSRStorage(
            len(ports[0].data),
            description="The data to write to the chosen lookup table.",
        )
        self.sel = CSRStorage(
            bits_for(len(ports) - 1),
            description="Write the index of the lookup table you want to write to to this CSR to trigger a write.",
        )

        # # #

        addr = Signal.like(self.addr.storage)
        data = Signal.like(self.data.storage)
        sel = Signal.like(self.sel.storage)
        en = Signal()

        self.specials += MultiReg(self.addr.storage, addr, "core")
        self.specials += MultiReg(self.data.storage, data, "core")
        self.specials += MultiReg(self.sel.storage, sel, "core")
        self.en_sync = PulseSynchronizer("sys", "core")
        self.comb += self.en_sync.i.eq(self.sel.re)
        self.comb += en.eq(self.en_sync.o)

        for i, port in enumerate(ports):
            self.comb += port.addr.eq(addr)
            self.comb += port.data.eq(data)
            self.comb += port.en.eq(en & (sel == i))


class CoreManager(LiteXModule):
    """
    A wrapper around one or more `Core`s that provides a CSR-based interface to
    them.
    """

    def __init__(
        self,
        platform,
        cuboids: int,
        max_area: int,
        cores: int,
        with_analyzer: bool = False,
    ):
        self.cores = CoreGroup(platform, cuboids, max_area, cores)

        area_bits = bits_for(max_area)
        square_bits = bits_for(max_area - 1)
        cursor_bits = square_bits + 2
        mapping_bits = cuboids * cursor_bits
        class_bits = bits_for(max_area - 1)
        mapping_index_bits = (cuboids - 1) * class_bits
        decision_index_bits = bits_for(max_decisions_len(max_area) - 1)

        prefix_bits = (
            area_bits + mapping_bits + mapping_index_bits + decision_index_bits
        )
        finder_bits = prefix_bits + max_decisions_len(max_area)

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

        self.neighbour_lookups = LookupManager(self.cores.neighbour_lookups)
        self.undo_lookups = LookupManager(self.cores.undo_lookups)

        self.active = CSRStatus(
            description="Whether there's still any work left for the core to do."
        )

        self.req_pause = CSRStorage(
            description="Set this CSR to 1 to tell all the cores to stop what they're doing and output their current states."
        )

        self.split_finders = CSRStatus(
            32,
            description="The number of new finders that have been produced via. splitting.",
        )
        self.completed_finders = CSRStatus(
            32, description="The total number of finders that have finished running."
        )

        for state in State.__members__:
            setattr(
                self,
                f"{state.lower()}_count",
                CSRStatus(
                    64,
                    # Seems like LiteX can't infer this when using `setattr`.
                    name=f"{state.lower()}_count",
                    description=f"The total number of clock cycles that cores have spent in {state} state.",
                ),
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
        self.specials += MultiReg(self.req_pause.storage, self.cores.req_pause, "core")

        self.split_finders_sync = BusSynchronizer(32, "core", "sys")
        self.comb += self.split_finders_sync.i.eq(self.cores.split_finders)
        self.comb += self.split_finders.status.eq(self.split_finders_sync.o)

        self.completed_finders_sync = BusSynchronizer(32, "core", "sys")
        self.comb += self.completed_finders_sync.i.eq(self.cores.completed_finders)
        self.comb += self.completed_finders.status.eq(self.completed_finders_sync.o)

        for state in State.__members__:
            csr = getattr(self, f"{state.lower()}_count")
            counter = getattr(self.cores, f"{state.lower()}_count")
            sync = BusSynchronizer(64, "core", "sys")
            setattr(self, f"{state.lower()}_count_sync", sync)
            self.comb += sync.i.eq(counter)
            self.comb += csr.status.eq(sync.o)

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
            self.analyzer = LiteScopeAnalyzer(
                analyzer_signals,
                depth=512,
                clock_domain="core",
                csr_csv="core_mgr_analyzer.csv",
            )
