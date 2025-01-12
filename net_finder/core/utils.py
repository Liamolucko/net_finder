from amaranth import *
from amaranth.hdl import ShapeLike, ValueLike
from amaranth.lib import stream, wiring
from amaranth.lib.wiring import In, Out


def pipe(m: Module, input: ValueLike, **kwargs) -> Signal:
    # src_loc_at tells Signal how far up in the call chain to look for what to name
    # the signal: so, setting it to 1 means we want it to use the name of the
    # variable the caller's assigning our result to.
    output = Signal.like(input, src_loc_at=1, **kwargs)
    m.d.sync += output.eq(input)
    return output


class SafeDemux(wiring.Component):
    """
    A component that lets you choose which of its sinks you want to connect its
    source to at any given time, and that prevents you from accidentally switching
    between them while sending a packet, or after asserting `valid`.
    """

    def __init__(self, payload_shape: ShapeLike, n: int):
        super().__init__(
            {
                "sink": In(stream.Signature(payload_shape)),
                "sources": Out(stream.Signature(payload_shape)).array(n),
                # The index of the source you want to output on.
                "sel": In(range(n)),
                # Whether you're okay with the current value of `sel` being locked in as the one
                # to receive the next packet.
                "en": In(1),
            }
        )

    def elaborate(self, platform) -> Module:
        m = Module()

        active = Signal()

        # The version of `self.sel` that gets locked to its current value during a
        # packet.
        sel = Signal.like(self.sel)

        for i, source in enumerate(self.sources):
            m.d.comb += source.payload.eq(self.sink.payload)
            m.d.comb += source.valid.eq(active & (sel == i) & self.sink.valid)

        m.d.comb += self.sink.ready.eq(active & Array(self.sources)[sel].ready)

        packet_end = self.sink.valid & self.sink.ready & self.sink.p.last

        # If the input's valid, we need to immediately lock in `sel` so that we don't
        # break the rule that `valid` stays high until a transaction happens; then we
        # deassert it when the last transaction in the packet takes place.
        m.d.sync += active.eq((active | (self.sink.valid & self.en)) & ~packet_end)
        with m.If(~active | packet_end):
            m.d.sync += sel.eq(self.sel)

        return m


class Merge(wiring.Component):
    """
    Receives packets from multiple streams and sends them all out over a single
    stream.
    """

    def __init__(self, payload_shape: ShapeLike, n: int):
        super().__init__(
            {
                "sinks": In(stream.Signature(payload_shape)).array(n),
                "source": Out(stream.Signature(payload_shape)),
            }
        )

    def elaborate(self, platform) -> Module:
        m = Module()

        sel = Signal(range(len(self.sinks)))
        active = Signal()

        for i, sink in enumerate(self.sinks):
            m.d.comb += sink.ready.eq(active & (sel == i) & self.source.ready)

        m.d.comb += self.source.valid.eq(active & Array(self.sinks)[sel].valid)
        m.d.comb += self.source.payload.eq(Array(self.sinks)[sel].payload)

        packet_end = self.source.valid & self.source.ready & self.source.p.last
        candidates = Signal(len(self.sinks))
        m.d.comb += candidates.eq(
            Cat(
                sink.valid & ~(active & (sel == i)) for i, sink in enumerate(self.sinks)
            )
        )

        m.d.sync += active.eq((active & ~packet_end) | candidates.any())
        with m.If(~active | packet_end):
            for i, is_candidate in enumerate(candidates):
                with m.If(is_candidate):
                    m.d.sync += sel.eq(i)

        return m


# Yoinked and adapted from LiteX.
class PipeValid(wiring.Component):
    """Pipe valid/payload to cut timing path"""

    def __init__(self, payload_shape: ShapeLike):
        super().__init__(
            {
                "sink": In(stream.Signature(payload_shape)),
                "source": Out(stream.Signature(payload_shape)),
            }
        )

    def elaborate(self, platform) -> Module:
        m = Module()

        with m.If(~self.source.valid | self.source.ready):
            # Pipe when source is not valid or is ready.
            m.d.sync += self.source.valid.eq(self.sink.valid)
            m.d.sync += self.source.payload.eq(self.sink.payload)
        m.d.comb += self.sink.ready.eq(~self.source.valid | self.source.ready)

        return m


class PipeReady(wiring.Component):
    """Pipe ready to cut timing path"""

    def __init__(self, payload_shape: ShapeLike):
        super().__init__(
            {
                "sink": In(stream.Signature(payload_shape)),
                "source": Out(stream.Signature(payload_shape)),
            }
        )

    def elaborate(self, platform) -> Module:
        m = Module()

        # Whether or not there's a piece of information buffered up in `payload`.
        valid = Signal()
        payload = Signal.like(self.sink.payload)

        # We'll be storing something next cycle if there's something to be transferred
        # which isn't going to be accepted.
        m.d.sync += valid.eq((valid | self.sink.valid) & ~self.source.ready)
        with m.If(~valid):
            m.d.sync += payload.eq(self.sink.payload)

        m.d.comb += self.sink.ready.eq(~valid)
        m.d.comb += self.source.valid.eq(valid | self.sink.valid)
        m.d.comb += self.source.payload.eq(Mux(valid, payload, self.sink.payload))

        return m
