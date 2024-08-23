from amaranth import *
from amaranth.lib import wiring
from amaranth.lib.data import ArrayLayout
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from .base_types import PosLayout, PosView, net_size
from .memory import ChunkedMemory
from .utils import pipe


def neighbour_shards(m: Module, pos: PosView):
    """Returns which net shards all of `pos`'s neighbours fall into."""

    output = Signal(ArrayLayout(2, 4), src_loc_at=1)

    with m.Switch(Cat(pos.x[:1], pos.y[:1])):
        # Each 4x4 chunk of the net is split up into shards like this:
        #     2301
        #     2301
        #     0123
        #     0123
        # where each number is the shard that that square is stored in. This layout
        # guarantees that for every square, all its neighbours are in different net
        # shards.
        with m.Case(0b0000, 0b1010):
            m.d.comb += output.eq([3, 0, 1, 2])
        with m.Case(0b0100, 0b1110):
            m.d.comb += output.eq([3, 2, 1, 0])
        with m.Case(0b0001, 0b1011):
            m.d.comb += output.eq([0, 1, 2, 3])
        with m.Case(0b0101, 0b1111):
            m.d.comb += output.eq([0, 3, 2, 1])
        with m.Case(0b0010, 0b1000):
            m.d.comb += output.eq([1, 2, 3, 0])
        with m.Case(0b0110, 0b1100):
            m.d.comb += output.eq([1, 0, 3, 2])
        with m.Case(0b0011, 0b1001):
            m.d.comb += output.eq([2, 3, 0, 1])
        with m.Case(0b0111, 0b1101):
            m.d.comb += output.eq([2, 1, 0, 3])

    return output


def shard_neighbours(m: Module, pos: PosView):
    """Returns which neighbour of `pos` each shard contains."""

    output = Signal(ArrayLayout(2, 4), src_loc_at=1)

    with m.Switch(Cat(pos.x[:1], pos.y[:1])):
        with m.Case(0b0000, 0b1010):
            m.d.comb += output.eq([1, 2, 3, 0])
        with m.Case(0b0100, 0b1110):
            m.d.comb += output.eq([3, 2, 1, 0])
        with m.Case(0b0001, 0b1011):
            m.d.comb += output.eq([0, 1, 2, 3])
        with m.Case(0b0101, 0b1111):
            m.d.comb += output.eq([0, 3, 2, 1])
        with m.Case(0b0010, 0b1000):
            m.d.comb += output.eq([3, 0, 1, 2])
        with m.Case(0b0110, 0b1100):
            m.d.comb += output.eq([1, 0, 3, 2])
        with m.Case(0b0011, 0b1001):
            m.d.comb += output.eq([2, 3, 0, 1])
        with m.Case(0b0111, 0b1101):
            m.d.comb += output.eq([2, 1, 0, 3])

    return output


class NetReadPortSignature(wiring.Signature):
    def __init__(self, *, max_area: int, chunk_width: int):
        super().__init__(
            {
                # I'm not bothering with `en`, I don't need it.
                "chunk": In(chunk_width),
                # The positions to read from the net: they must be the neighbours of `pos` in
                # the usual left, up, right, down order.
                "neighbours": In(PosLayout(max_area)).array(4),
                # The position that `neighbours` contains the neighbours of: this allows
                # computing which shard they fall into more easily.
                "pos": In(PosLayout(max_area)),
                "data": Out(1).array(4),
            }
        )


class NetWritePortSignature(wiring.Signature):
    def __init__(self, *, max_area: int, chunk_width: int):
        super().__init__(
            {
                "en": In(1),
                "chunk": In(chunk_width),
                # The positions to read from the net: they must be the neighbours of `pos` in
                # the usual left, up, right, down order.
                "neighbours": In(PosLayout(max_area)).array(4),
                # The position that `neighbours` contains the neighbours of: this allows
                # computing which shard they fall into more easily.
                "pos": In(PosLayout(max_area)),
                "data": In(1).array(4),
            }
        )


class Net(wiring.Component):
    """
    Stores whether each net position has an instruction queued to set it.

    It provides a simple-dual-port interface, where each port can access the 4
    neighbours of a position each clock cycle; however, it assumes that the read and
    write port are always accessing different chunks.
    """

    def __init__(self, max_area: int, chunks: int):
        super().__init__(
            {
                "read_port": Out(
                    NetReadPortSignature(
                        max_area=max_area, chunk_width=ceil_log2(chunks)
                    )
                ),
                "write_port": Out(
                    NetWritePortSignature(
                        max_area=max_area, chunk_width=ceil_log2(chunks)
                    )
                ),
            }
        )

        self._max_area = max_area
        self._chunks = chunks

    def elaborate(self, platform) -> Module:
        m = Module()

        read_neighbour_shards = neighbour_shards(m, self.read_port.pos)
        read_shard_neighbours = shard_neighbours(m, self.read_port.pos)
        write_shard_neighbours = shard_neighbours(m, self.write_port.pos)

        inner_read_ports = []
        for i in range(4):
            net_size_ = net_size(self._max_area)
            shard = ChunkedMemory(
                shape=1,
                # We need to round one of the dimensions up to the next power of two in order
                # for concatenating the x and y coordinates to work properly.
                depth=(net_size_ << ceil_log2(net_size_)) // 4,
                chunks=self._chunks,
            )
            m.submodules[f"shard{i}"] = shard

            inner_read_port, inner_write_port = shard.sdp_port()

            neighbour = self.read_port.neighbours[read_shard_neighbours[i]]
            m.d.comb += inner_read_port.chunk.eq(self.read_port.chunk)
            m.d.comb += inner_read_port.addr.eq(Cat(neighbour.x[2:], neighbour.y))

            neighbour_index = write_shard_neighbours[i]
            neighbour = self.write_port.neighbours[neighbour_index]
            m.d.comb += inner_write_port.en.eq(self.write_port.en)
            m.d.comb += inner_write_port.chunk.eq(self.write_port.chunk)
            m.d.comb += inner_write_port.addr.eq(Cat(neighbour.x[2:], neighbour.y))
            m.d.comb += inner_write_port.data.eq(self.write_port.data[neighbour_index])

            inner_read_ports.append(inner_read_port)

        prev_read_neighbour_shards = [pipe(m, shard) for shard in read_neighbour_shards]

        for i in range(4):
            shard = prev_read_neighbour_shards[i]
            m.d.comb += self.read_port.data[i].eq(inner_read_ports[shard].data)

        return m
