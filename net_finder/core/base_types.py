import math

from amaranth import *
from amaranth.hdl import ValueLike
from amaranth.lib import data
from amaranth.utils import ceil_log2


def net_size(max_area: int):
    """Returns the width/height of the net."""

    # Round it up to the nearest multiple of 4, since our scheme for splitting the
    # net up into 4 shards requires being able to divide it up into 4x4 tiles.
    return int(4 * math.ceil(max_area / 4))


class PosLayout(data.StructLayout):
    def __init__(self, max_area: int):
        super().__init__(
            {
                "x": ceil_log2(net_size(max_area)),
                "y": ceil_log2(net_size(max_area)),
            }
        )

        self._net_size = net_size(max_area)

    def __call__(self, value):
        return PosView(self, value)


class PosView(data.View):
    def moved_in(self, m: Module, direction: ValueLike):
        # src_loc_at tells Signal how far up in the call chain to look for what to name
        # the signal: so, setting it to 1 means we want it to use the name of the
        # variable the caller's assigning our result to.
        output = Signal(self.shape(), src_loc_at=1)

        unchecked_x = Signal(self.x.shape())
        unchecked_y = Signal(self.y.shape())

        with m.Switch(direction):
            with m.Case(0):
                m.d.comb += unchecked_x.eq(self.x - 1)
                m.d.comb += unchecked_y.eq(self.y)
            with m.Case(1):
                m.d.comb += unchecked_x.eq(self.x)
                m.d.comb += unchecked_y.eq(self.y + 1)
            with m.Case(2):
                m.d.comb += unchecked_x.eq(self.x + 1)
                m.d.comb += unchecked_y.eq(self.y)
            with m.Case(3):
                m.d.comb += unchecked_x.eq(self.x)
                m.d.comb += unchecked_y.eq(self.y - 1)

        # This probably isn't really necessary since `net_size` is now always a power of
        # 2 (due to `ChunkedMemory` requiring that), and the natural wrapping already
        # does what we want, but it doesn't hurt to include this in case that changes.
        #
        # It's done as a postprocessing step like this to make it extremely obvious to
        # the synthesiser that no extra logic is needed in the power-of-2 case: when it
        # sees unchecked_x == 0 ? 0 : unchecked_x == 0x3f ? 0x3f : unchecked_x, it can
        # easily optimise that down to just unchecked_x.
        net_size = self.shape()._net_size
        coord_underflow = (1 << self.x.shape().width) - 1
        coord_overflow = net_size & coord_underflow

        with m.Switch(unchecked_x):
            with m.Case(coord_overflow):
                m.d.comb += output.x.eq(0)
            with m.Case(coord_underflow):
                m.d.comb += output.x.eq(net_size - 1)
            with m.Default():
                m.d.comb += output.x.eq(unchecked_x)

        with m.Switch(unchecked_y):
            with m.Case(coord_overflow):
                m.d.comb += output.y.eq(0)
            with m.Case(coord_underflow):
                m.d.comb += output.y.eq(net_size - 1)
            with m.Default():
                m.d.comb += output.y.eq(unchecked_y)

        return output


def cursor_layout(max_area: int):
    return data.StructLayout(
        {
            "orientation": 2,
            "square": ceil_log2(max_area),
        }
    )


def mapping_layout(cuboids: int, max_area: int):
    return data.ArrayLayout(cursor_layout(max_area), cuboids)


def instruction_layout(cuboids: int, max_area: int):
    return data.StructLayout(
        {
            "pos": PosLayout(max_area),
            "mapping": mapping_layout(cuboids, max_area),
        }
    )
