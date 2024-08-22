from litex.gen import *
from litex.soc.interconnect import stream
from migen.genlib.coding import PriorityEncoder


def varlen_layout(width: int, param_layout=[]):
    return stream.EndpointDescription(
        [("data", width), ("len", bits_for(width))], param_layout
    )


def flat_layout(param_layout=[]):
    return stream.EndpointDescription([("data", 1)], param_layout)


class Flatten(LiteXModule):
    """
    A converter which takes in a stream of variable-length payloads, and outputs
    them as a stream of 1-bit payloads (from MSB to LSB).
    """

    def __init__(self, width: int, param_layout=[]):
        self.sink = stream.Endpoint(varlen_layout(width, param_layout))
        self.source = stream.Endpoint(flat_layout(param_layout))

        # # #

        # How many bits of `self.sink.data` we still have to emit.
        bits_left = Signal(max=width + 1)

        # The least significant bit of `self.sink.data` that we have to emit.
        base_bit = width - self.sink.len

        cases = {i: self.source.data.eq(self.sink.data[i]) for i in range(width)}
        self.comb += Case(base_bit + bits_left - 1, cases)

        self.comb += self.source.param.eq(self.sink.param)
        self.comb += self.source.valid.eq(bits_left != 0)
        self.comb += self.source.first.eq(
            self.sink.first & (bits_left == self.sink.len)
        )
        self.comb += self.source.last.eq(self.sink.last & (bits_left == 1))

        self.comb += self.sink.ready.eq(self.source.ready & (bits_left == 1))

        self.sync += If(
            self.sink.valid & (bits_left == 0),
            # If `bits_left` is 0, that means we're finished with the previous packet, and
            # so this must be the start of a new packet.
            bits_left.eq(self.sink.len),
        ).Elif(
            self.source.valid & self.source.ready,
            bits_left.eq(bits_left - 1),
        )


class Collect(LiteXModule):
    """
    A converter which takes in a stream of 1-bit payloads, and collects them
    together into `width`-bit payloads from MSB to LSB (or, if the packet cuts off
    before `width` is reached, variable-length payloads).

    When emitting variable-length payloads, the bits after the filled portion will
    be garbage.
    """

    def __init__(self, width: int, param_layout=[]):
        self.sink = stream.Endpoint(flat_layout(param_layout))
        self.source = stream.Endpoint(varlen_layout(width, param_layout))

        # # #

        # The bits we've collected so far.
        data = Signal(width)
        # How many bits we've collected into `data`.
        len = Signal(max=width + 1)
        # The parameters associated with the current packet.
        param = Record(param_layout)
        # Whether the first bit we collected as part of this payload was the first bit
        # in its packet.
        first = Signal()
        # Whether the last bit we collected as part of this payload was the last bit in
        # its packet.
        last = Signal()
        # Whether `data` is valid and ready to be emitted.
        done = Signal()

        self.sync += If(
            self.source.valid & self.source.ready,
            len.eq(0),
        ).Elif(
            self.sink.valid & self.sink.ready,
            len.eq(len + 1),
        )

        cases = {i: data[i].eq(self.sink.data) for i in range(width)}
        self.sync += If(
            self.sink.valid & self.sink.ready,
            Case(width - len - 1, cases),
            param.eq(self.sink.param),
        )

        self.sync += If(
            # If we receive the first bit of a packet, we set `first` to 1.
            self.sink.valid & self.sink.ready & self.sink.first,
            first.eq(1),
        ).Elif(
            # If we finish a payload and aren't immediately starting a new packet (covered
            # by the above branch), we reset `first` to 0.
            self.source.valid & self.source.ready,
            first.eq(0),
        )

        self.sync += If(
            # When we receive the last bit of a packet, we set `last` to 1.
            self.sink.valid & self.sink.ready & self.sink.last,
            last.eq(1),
        ).Elif(
            # Then we set it back to 0 once our payload is accepted.
            self.source.valid & self.source.ready,
            last.eq(0),
        )

        self.sync += If(
            self.sink.valid & self.sink.ready & (self.sink.last | (len == width - 1)),
            done.eq(1),
        ).Elif(self.source.valid & self.source.ready, done.eq(0))

        self.comb += self.sink.ready.eq(~done)

        self.comb += self.source.data.eq(data)
        self.comb += self.source.len.eq(len)
        self.comb += self.source.param.eq(param)
        self.comb += self.source.valid.eq(done)
        self.comb += self.source.first.eq(first)
        self.comb += self.source.last.eq(last)


class SafeMux(LiteXModule):
    """
    A wrapper around a `stream.Multiplexer` that prevents you from accidentally
    changing `sel` while receiving a packet.
    """

    def __init__(self, layout, n: int):
        self.mux = stream.Multiplexer(layout, n)

        for i in range(n):
            setattr(self, f"sink{i}", getattr(self.mux, f"sink{i}"))
        self.source = self.mux.source
        self.sel = Signal.like(self.mux.sel)

        # # #

        # Whether a packet is currently being received through this mux.
        active = Signal()
        self.sync += If(
            self.source.valid & self.source.ready & self.source.first,
            active.eq(1),
        ).Elif(
            self.source.valid & self.source.ready & self.source.last,
            active.eq(0),
        )

        # What `self.mux.sel` was during the previous clock cycle.
        prev_sel = Signal.like(self.mux.sel)
        self.sync += prev_sel.eq(self.mux.sel)
        self.comb += If(active, self.mux.sel.eq(prev_sel)).Else(
            self.mux.sel.eq(self.sel)
        )


class SafeDemux(LiteXModule):
    """
    A wrapper around a `stream.Demultiplexer` that prevents you from accidentally
    changing `sel` while sending a packet, or after asserting `valid`.
    """

    def __init__(self, layout, n: int):
        self.mux = stream.Demultiplexer(layout, n)

        self.sink = self.mux.sink
        for i in range(n):
            setattr(self, f"source{i}", getattr(self.mux, f"source{i}"))
        self.sel = Signal.like(self.mux.sel)

        # Whether a packet is currently being sent through this demux.
        self.active = Signal()

        # # #

        self.sync += If(
            self.sink.valid,
            # If the input's valid, we need to immediately lock in `sel` so that we don't
            # break the rule that `valid` stays high until a transaction happens; then we
            # deassert it when the last transaction in the packet takes place.
            self.active.eq(~(self.sink.ready & self.sink.last)),
        )

        # What `self.mux.sel` was during the previous clock cycle.
        prev_sel = Signal.like(self.mux.sel)
        self.sync += prev_sel.eq(self.mux.sel)
        self.comb += If(
            self.active,
            self.mux.sel.eq(prev_sel),
        ).Else(
            self.mux.sel.eq(self.sel),
        )


class Merge(LiteXModule):
    """
    Receives packets from multiple streams and sends them all out over a single
    stream.
    """

    def __init__(self, layout, n: int):
        self.mux = SafeMux(layout, n)

        for i in range(n):
            setattr(self, f"sink{i}", getattr(self.mux, f"sink{i}"))
        self.source = self.mux.source

        # # #

        self.enc = PriorityEncoder(n)
        self.comb += self.enc.i.eq(
            Cat(getattr(self, f"sink{i}").valid for i in range(n))
        )
        # Make this synchronous so that one core's `out_valid` doesn't affect another
        # core's `out_ready`, creating a timing path that spans the entire FPGA.
        self.sync += self.mux.sel.eq(self.enc.o)
        # It's fine to ignore `enc.n`, `enc.o` just defaults to 0 when no bits are set
        # and then nothing will happen since `self.sink0.valid` isn't set.
