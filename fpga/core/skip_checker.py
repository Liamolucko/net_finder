from amaranth import *
from amaranth.lib import data, wiring
from amaranth.lib.memory import MemoryData, ReadPort
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from .base_types import mapping_layout
from .utils import pipe


def undo_lookup_entry_layout(max_area: int):
    return data.StructLayout(
        {
            # The lower 3 bits of the class you get by undoing the given transform on the
            # given cursor's class.
            "lo0": 3,
            # If the fixed class isn't affected by flipping, the lower 3 bits of the class
            # you get by undoing the given transform with its MSB (flip bit) set to 1.
            #
            # Otherwise, this is the same as lo0.
            #
            # Normally, we rely on the fact that the fixed class is the one in its family
            # with transform 0 to make sure that once we've transformed a class mapping into
            # having the fixed class, we can directly compute its index. However, when the
            # fixed class is unaffected by flipping, this breaks down since you can flip a
            # class mapping to get an equivalent mapping which still uses the fixed class,
            # and either one of them could be the smaller one we need to compute the index
            # from.
            #
            # As a result, we need to store both `lo0` and `lo1` so that we can get both
            # versions of the mapping in that scenario and see which is smaller.
            "lo1": 3,
            # The upper bits of the class you get by undoing the given transform on the
            # given cursor's class.
            "hi": ceil_log2(max_area) - 3,
        }
    )


def undo_lookup_layout(max_area: int):
    return MemoryData(
        shape=undo_lookup_entry_layout(max_area),
        # We need an entry for every combination of cursor + transform to undo.
        depth=(4 * max_area) * 8,
        init=[],
    )


class SkipChecker(wiring.Component):
    def __init__(self, cuboids: int, max_area: int):
        ul_layout = undo_lookup_layout(max_area)

        super().__init__(
            {
                # The ports through which the undo lookups should be accessed.
                "ports": In(
                    ReadPort.Signature(
                        addr_width=ceil_log2(ul_layout.depth), shape=ul_layout.shape
                    )
                ).array(cuboids - 1),
                # The index of the mapping we started searching from.
                "start_mapping_index": In(cuboids * ceil_log2(max_area)),
                # The mapping to check.
                "input": In(mapping_layout(cuboids, max_area)),
                # Whether `input[0]` is in the fixed family.
                "fixed_family": In(1),
                # The transform of `input[0]`'s class.
                "transform": In(3),
                # Whether `input` should be skipped.
                "skip": Out(1),
            }
        )

        self._cuboids = cuboids
        self._max_area = max_area

    def elaborate(self, platform) -> Module:
        m = Module()

        for i in range(1, self._cuboids):
            m.d.comb += self.ports[i].addr.eq(8 * self.input[0] + self.transform)

        prev_start_mapping_index = pipe(m, self.start_mapping_index)
        prev_fixed_family = pipe(m, self.fixed_family)

        index0 = Cat(Cat(port.data.lo0, port.data.hi) for port in self.ports)
        index1 = Cat(Cat(port.data.lo1, port.data.hi) for port in self.ports)

        m.d.comb += self.skip.eq(
            prev_fixed_family
            & (
                (index0 < prev_start_mapping_index)
                | (index1 < prev_start_mapping_index)
            )
        )

        return m
