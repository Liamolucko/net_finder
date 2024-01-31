from litex.gen import *

from litex.soc.interconnect import stream


class Flatten(Module):
    """
    A converter which takes in a stream of variable-length payloads, and outputs
    them as a stream of 1-bit payloads (from MSB to LSB).
    """

    def __init__(self, width: int):
        self.sink = stream.Endpoint([("data", width), ("len", bits_for(width))])
        self.source = stream.Endpoint([("data", 1)])

        # # #

        # How many bits of `self.sink.data` we still have to emit.
        bits_left = Signal(max=width + 1)

        # The least significant bit of `self.sink.data` that we have to emit.
        base_bit = width - self.sink.len

        cases = {i: self.source.data.eq(self.sink.data[i]) for i in range(width)}
        self.comb += Case(base_bit + bits_left - 1, cases)

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
        ).Elif(self.source.valid & self.source.ready, bits_left.eq(bits_left - 1))


class Collect(Module):
    """
    A converter which takes in a stream of 1-bit payloads, and collects them
    together into `width`-bit payloads from MSB to LSB (or, if the packet cuts off
    before `width` is reached, variable-length payloads).

    When emitting variable-length payloads, the bits after the filled portion will
    be garbage.
    """

    def __init__(self, width: int):
        self.sink = stream.Endpoint([("data", 1)])
        self.source = stream.Endpoint([("data", width), ("len", bits_for(width))])

        # # #

        # The bits we've collected so far.
        data = Signal(width)
        # How many bits we've collected into `data`.
        len = Signal(max=width + 1)
        # Whether the first bit we collected as part of this payload was the first bit
        # in its packet.
        first = Signal()
        # Whether the last bit we collected as part of this payload was the last bit in
        # its packet.
        last = Signal()
        # Whether `data` is valid and ready to be emitted.
        done = Signal()

        self.sync += If(self.source.valid & self.source.ready, len.eq(0)).Elif(
            self.sink.valid & self.sink.ready, len.eq(len + 1)
        )

        cases = {i: data[i].eq(self.sink.data) for i in range(width)}
        self.sync += If(self.sink.valid & self.sink.ready, Case(width - len - 1, cases))

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
        self.comb += self.source.valid.eq(done)
        self.comb += self.source.first.eq(first)
        self.comb += self.source.last.eq(last)
