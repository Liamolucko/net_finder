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
            # Let's call the class in the fixed family with the given transform `c`.
            #
            # Sometimes, there are multiple transformations that always have the same effect
            # on a family: if this is the case for the fixed family, we can undo any of
            # the equivalent transforms that take you from the fixed class to `c`.
            #
            # These transforms don't necessarily do the same thing to other families though:
            # as a result, we need to try undoing _all_ of the transforms that are
            # equivalent to the given transform as far as the fixed family is concerned.
            #
            # So, this is a list of the lower 3 bits of the classes you get by undoing each
            # of those transforms on the given cursor's class. If there are less than 2
            # equivalent transforms, both entries are the same.
            #
            # The actual number of equivalent transforms is 2^(3 - transform bits); and, at
            # least up to area 64, the fixed classes always have at least 2 transform bits,
            # so we only need to worry about 2 different possibilities.
            "lo": data.ArrayLayout(3, 2),
            # The upper bits of the given cursor's class (i.e., the ones that stay the same
            # regardless of transformation).
            "hi": ceil_log2(max_area) - 3,
        }
    )


def undo_lookup_layout(max_area: int, init=[]):
    return MemoryData(
        shape=undo_lookup_entry_layout(max_area),
        # We need an entry for every combination of cursor + transform to undo.
        depth=(4 * max_area) * 8,
        init=init,
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
                "start_mapping_index": In((cuboids - 1) * ceil_log2(max_area)),
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
            m.d.comb += self.ports[i - 1].addr.eq(
                8 * self.input[i].as_value() + self.transform
            )

        prev_start_mapping_index = pipe(m, self.start_mapping_index)
        prev_fixed_family = pipe(m, self.fixed_family)

        indices = [
            Cat(Cat(port.data.lo[i], port.data.hi) for port in self.ports[::-1])
            for i in range(2)
        ]

        m.d.comb += self.skip.eq(
            prev_fixed_family
            & Cat(index < prev_start_mapping_index for index in indices).any()
        )

        return m
