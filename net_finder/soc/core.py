from os import path

from amaranth.back import verilog
from litescope import LiteScopeAnalyzer
from litex.gen import *
from litex.soc.interconnect import stream
from litex.soc.interconnect.csr import *
from migen.genlib.fifo import SyncFIFOBuffered

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
        self.fifo_has_room = Signal()

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

        FINDERS_PER_CORE = 4
        self.output_merge_sel = Signal(bits_for(n * FINDERS_PER_CORE - 1))
        self.output_merge_active = Signal(1)
        self.output_merge_source_valid = Signal(1)
        self.output_merge_source_ready = Signal(1)
        self.output_merge_source_payload = Signal(4)
        self.output_mux_active = Signal(1)
        self.output_mux_latched_sel = Signal(1)
        self.loopback_valid = Signal(1)
        self.loopback_ready = Signal(1)
        self.loopback_payload = Signal(2)
        self.input_merge_sel = Signal(1)
        self.input_merge_active = Signal(1)
        self.input_mux_active = Signal(1)
        self.input_mux_latched_sel = Signal(bits_for(n * FINDERS_PER_CORE - 1))
        self.input_mux_sink_valid = Signal(1)
        self.input_mux_sink_ready = Signal(1)
        self.input_mux_sink_payload = Signal(2)
        self.splittable_base = Signal(bits_for(max_decisions_len(max_area)))
        self.splittee = Signal(bits_for(n * FINDERS_PER_CORE - 1))
        self.split_wanted = Signal(1)
        self.splittee_sink_valid = Signal(1)
        self.splittee_sink_ready = Signal(1)
        self.splittee_sink_payload = Signal(2)
        self.splittee_source_valid = Signal(1)
        self.splittee_source_ready = Signal(1)
        self.splittee_source_payload = Signal(4)
        self.splittee_req_pause = Signal(1)
        self.splittee_req_split = Signal(1)
        self.splittee_wants_finder = Signal(1)
        self.splittee_state = Signal(3)
        self.splittee_base_decision = Signal(bits_for(max_decisions_len(max_area)))
        self.sender_sink_valid = Signal(1)
        self.sender_sink_ready = Signal(1)
        self.sender_sink_payload = Signal(2)
        self.sender_source_valid = Signal(1)
        self.sender_source_ready = Signal(1)
        self.sender_source_payload = Signal(4)
        self.sender_req_pause = Signal(1)
        self.sender_req_split = Signal(1)
        self.sender_wants_finder = Signal(1)
        self.sender_state = Signal(3)
        self.sender_base_decision = Signal(bits_for(max_decisions_len(max_area)))
        self.receiver_sink_valid = Signal(1)
        self.receiver_sink_ready = Signal(1)
        self.receiver_sink_payload = Signal(2)
        self.receiver_source_valid = Signal(1)
        self.receiver_source_ready = Signal(1)
        self.receiver_source_payload = Signal(4)
        self.receiver_req_pause = Signal(1)
        self.receiver_req_split = Signal(1)
        self.receiver_wants_finder = Signal(1)
        self.receiver_state = Signal(3)
        self.receiver_base_decision = Signal(bits_for(max_decisions_len(max_area)))

        # # #

        raw_sink_payload = Record(CORE_IN_LAYOUT)
        raw_source_payload = Record(CORE_GROUP_OUT_LAYOUT)

        self.specials += Instance(
            "core_group",
            i_core_clk=ClockSignal("core"),
            i_core_rst=ResetSignal("core"),
            i_sys_clk=ClockSignal("sys"),
            i_sys_rst=ResetSignal("sys"),
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
            i_fifo_has_room=self.fifo_has_room,
            i_req_pause=self.req_pause,
            o_active=self.active,
            o_split_finders=self.split_finders,
            o_completed_finders=self.completed_finders,
            **{
                f"o_{state.lower()}_count": getattr(self, f"{state.lower()}_count")
                for state in State.__members__
            },
            o_output_merge_sel=self.output_merge_sel,
            o_output_merge_active=self.output_merge_active,
            o_output_merge_source_valid=self.output_merge_source_valid,
            o_output_merge_source_ready=self.output_merge_source_ready,
            o_output_merge_source_payload=self.output_merge_source_payload,
            o_output_mux_active=self.output_mux_active,
            o_output_mux_latched_sel=self.output_mux_latched_sel,
            o_loopback_valid=self.loopback_valid,
            o_loopback_ready=self.loopback_ready,
            o_loopback_payload=self.loopback_payload,
            o_input_merge_sel=self.input_merge_sel,
            o_input_merge_active=self.input_merge_active,
            o_input_mux_active=self.input_mux_active,
            o_input_mux_latched_sel=self.input_mux_latched_sel,
            o_input_mux_sink_valid=self.input_mux_sink_valid,
            o_input_mux_sink_ready=self.input_mux_sink_ready,
            o_input_mux_sink_payload=self.input_mux_sink_payload,
            o_splittable_base=self.splittable_base,
            o_splittee=self.splittee,
            o_split_wanted=self.split_wanted,
            o_splittee_sink_valid=self.splittee_sink_valid,
            o_splittee_sink_ready=self.splittee_sink_ready,
            o_splittee_sink_payload=self.splittee_sink_payload,
            o_splittee_source_valid=self.splittee_source_valid,
            o_splittee_source_ready=self.splittee_source_ready,
            o_splittee_source_payload=self.splittee_source_payload,
            o_splittee_req_pause=self.splittee_req_pause,
            o_splittee_req_split=self.splittee_req_split,
            o_splittee_wants_finder=self.splittee_wants_finder,
            o_splittee_state=self.splittee_state,
            o_splittee_base_decision=self.splittee_base_decision,
            o_sender_sink_valid=self.sender_sink_valid,
            o_sender_sink_ready=self.sender_sink_ready,
            o_sender_sink_payload=self.sender_sink_payload,
            o_sender_source_valid=self.sender_source_valid,
            o_sender_source_ready=self.sender_source_ready,
            o_sender_source_payload=self.sender_source_payload,
            o_sender_req_pause=self.sender_req_pause,
            o_sender_req_split=self.sender_req_split,
            o_sender_wants_finder=self.sender_wants_finder,
            o_sender_state=self.sender_state,
            o_sender_base_decision=self.sender_base_decision,
            o_receiver_sink_valid=self.receiver_sink_valid,
            o_receiver_sink_ready=self.receiver_sink_ready,
            o_receiver_sink_payload=self.receiver_sink_payload,
            o_receiver_source_valid=self.receiver_source_valid,
            o_receiver_source_ready=self.receiver_source_ready,
            o_receiver_source_payload=self.receiver_source_payload,
            o_receiver_req_pause=self.receiver_req_pause,
            o_receiver_req_split=self.receiver_req_split,
            o_receiver_wants_finder=self.receiver_wants_finder,
            o_receiver_state=self.receiver_state,
            o_receiver_base_decision=self.receiver_base_decision,
        )

        self.comb += raw_sink_payload.data.eq(self.sink.data)
        self.comb += raw_sink_payload.last.eq(self.sink.last)

        self.comb += self.source.data.eq(raw_source_payload.data)
        self.comb += self.source.last.eq(raw_source_payload.last)
        self.comb += self.source.kind.eq(raw_source_payload.kind)

        out_first = Signal(reset=1)
        self.comb += self.source.first.eq(out_first)
        self.sync.sys += If(
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

        for i, port in enumerate(ports):
            self.comb += port.addr.eq(self.addr.storage)
            self.comb += port.data.eq(self.data.storage)
            self.comb += port.en.eq(self.sel.re & (self.sel.storage == i))


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
        input_fifo = SyncFIFOBuffered(self.input.size, fifo_depth)
        self.input_fifo = ClockDomainsRenamer("sys")(input_fifo)

        self.comb += self.input_fifo.din.eq(self.input.storage)
        self.comb += self.flow.fields.input_ready.eq(self.input_fifo.writable)
        self.comb += self.input_fifo.we.eq(self.input_submit.re)

        input_conv = Flatten(finder_bits)
        self.input_conv = ClockDomainsRenamer("sys")(input_conv)
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
        output_fifo = SyncFIFOBuffered(self.output.size, fifo_depth)
        self.output_fifo = ClockDomainsRenamer("sys")(output_fifo)

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

        self.output_conv = ClockDomainsRenamer("sys")(
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

        self.comb += self.cores.fifo_has_room.eq(
            self.output_fifo.level <= fifo_depth - 2
        )

        self.comb += self.active.status.eq(self.cores.active)
        self.comb += self.cores.req_pause.eq(self.req_pause.storage)
        self.comb += self.split_finders.status.eq(self.cores.split_finders)
        self.comb += self.completed_finders.status.eq(self.cores.completed_finders)

        for state in State.__members__:
            csr = getattr(self, f"{state.lower()}_count")
            counter = getattr(self.cores, f"{state.lower()}_count")
            self.comb += csr.status.eq(counter)

        # Analyzer ---------------------------------------------------------------------------------
        if with_analyzer:
            analyzer_signals = [
                self.cores.sink,
                self.cores.source,
                *(self.cores.neighbour_lookups[i] for i in range(cuboids)),
                *(self.cores.undo_lookups[i] for i in range(cuboids - 1)),
                self.cores.req_pause,
                self.cores.active,
                self.input_fifo.writable,
                self.input_fifo.we,
                self.input_fifo.readable,
                self.input_fifo.re,
                self.input_fifo.level,
                self.output_fifo.writable,
                self.output_fifo.we,
                self.output_fifo.readable,
                self.output_fifo.re,
                self.output_fifo.level,
                self.input_conv.sink,
                self.output_conv.source,
                self.cores.output_merge_sel,
                self.cores.output_merge_active,
                self.cores.output_merge_source_valid,
                self.cores.output_merge_source_ready,
                self.cores.output_merge_source_payload,
                self.cores.output_mux_active,
                self.cores.output_mux_latched_sel,
                self.cores.loopback_valid,
                self.cores.loopback_ready,
                self.cores.loopback_payload,
                self.cores.input_merge_sel,
                self.cores.input_merge_active,
                self.cores.input_mux_active,
                self.cores.input_mux_latched_sel,
                self.cores.input_mux_sink_valid,
                self.cores.input_mux_sink_ready,
                self.cores.input_mux_sink_payload,
                self.cores.splittable_base,
                self.cores.splittee,
                self.cores.split_wanted,
                self.cores.splittee_sink_valid,
                self.cores.splittee_sink_ready,
                self.cores.splittee_sink_payload,
                self.cores.splittee_source_valid,
                self.cores.splittee_source_ready,
                self.cores.splittee_source_payload,
                self.cores.splittee_req_pause,
                self.cores.splittee_req_split,
                self.cores.splittee_wants_finder,
                self.cores.splittee_state,
                self.cores.splittee_base_decision,
                self.cores.sender_sink_valid,
                self.cores.sender_sink_ready,
                self.cores.sender_sink_payload,
                self.cores.sender_source_valid,
                self.cores.sender_source_ready,
                self.cores.sender_source_payload,
                self.cores.sender_req_pause,
                self.cores.sender_req_split,
                self.cores.sender_wants_finder,
                self.cores.sender_state,
                self.cores.sender_base_decision,
                self.cores.receiver_sink_valid,
                self.cores.receiver_sink_ready,
                self.cores.receiver_sink_payload,
                self.cores.receiver_source_valid,
                self.cores.receiver_source_ready,
                self.cores.receiver_source_payload,
                self.cores.receiver_req_pause,
                self.cores.receiver_req_split,
                self.cores.receiver_wants_finder,
                self.cores.receiver_state,
                self.cores.receiver_base_decision,
            ]
            self.analyzer = LiteScopeAnalyzer(
                analyzer_signals,
                depth=512,
                clock_domain="sys",
                csr_csv="core_mgr_analyzer.csv",
            )
