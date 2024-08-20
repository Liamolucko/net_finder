from amaranth import *
from amaranth.hdl import ValueLike
from amaranth.lib import data, wiring
from amaranth.lib.memory import MemoryData, ReadPort, ceil_log2
from amaranth.lib.wiring import In, Out

from .base_types import PosLayout, cursor_layout, instruction_layout
from .utils import pipe


def opposite(direction):
    return (direction + 2)[:2]


class NeighbourLookupEntryLayout(data.StructLayout):
    def __init__(self, max_area: int):
        super().__init__(
            {
                # The actual neighbours; these can be either the regular neighbours or the
                # T-shaped neighbours depending on the address.
                #
                # The regular neighbours are what you'd expect: the 4 cursors adjacent to the
                # input square. So with an input square `s`, it looks like this:
                #
                # ```
                #  1
                # 0s2
                #  3
                # ```
                #
                # As indicated by the numbering, this array contains the neighbours in clockwise
                # order starting from the left one.
                #
                # Then, the T-shaped neighbours look like this:
                #
                # ```
                #  1
                # s02
                #  3
                # ```
                #
                # This makes sense to have because we only store the parents of the instructions
                # we want to run in order to save space, and so this allows getting both the
                # instructions themselves as well as all their neighbours at the same time.
                #
                # The diagram is oriented such that the direction is always right. If we orient
                # it relative to the orientation of the cursors, it can look like any of these:
                #
                # ```
                #  1   s   3   2
                # s02 301 20s 103
                #  3   2   1   s
                # ```
                #
                # `neighbours[0]` is always the middle, and then the rest go in clockwise order
                # starting from the input square.
                #
                # A previous layout I considered looked like this:
                #
                # ```
                #  1   s   1   1
                # s02 012 02s 032
                #  3   3   3   s
                # ```
                #
                # It seems like it might be beneficial because it only requires 2-to-1 muxes for
                # the neighbours, rather than 4-to-1 muxes: however, that actually isn't the
                # case, because we only take an input square, not the actual underlying input
                # cursor.
                #
                # This means that the neighbours need to be rotated around afterwards to line up
                # with the cursor's orientation, and thus require 4-to-1 muxes anyway. So, we
                # may as well get the benefit of the other design that the middle is always in
                # the same spot.
                "neighbours": data.ArrayLayout(cursor_layout(max_area), 4),
                # If this is the first cuboid, whether the centre of this layout is in the fixed
                # family.
                #
                # Otherwise, this is always 0.
                #
                # By 'the centre of this layout', I mean the input square when using the regular
                # neighbours, and the immediate neighbour of the input square when using the
                # T-shaped neighbours.
                #
                # It doesn't really make sense to talk about the class of a square, since
                # orientation can have an effect; however, that isn't the case for family, since
                # one of the transformations it allows is rotating all the cuboids in tandem
                # around a cursor, thus reaching all orientations of the starting cursor.
                "fixed_family": 1,
                # If `fixed_family` is 1, the transform of the centre's class.
                #
                # This can't be used as-is though, because the input is a square when we want to
                # get the transform of a cursor. However, rotating a cursor always leads to the
                # bottom 2 bits of its class's transform changing by the same amount, so you can
                # get to the real transform by adding on the cursor's orientation.
                #
                # If the cursor's class has less than 2 transform bits, it just means that the
                # rotations go 0 -> 1 -> 0 -> 1 (or 0 -> 0 -> 0 -> 0) rather than 00 -> 01 -> 10
                # -> 11. So addition then truncation works: we only support fixed classes with
                # at least 2 transform bits right now anyway though.
                #
                # That's also the reason why this is only valid if `fixed_family` is 1: if this
                # isn't the fixed family, we don't know how many bits it has and thus don't know
                # what to truncate it to.
                "transform": 3,
            }
        )

    def __call__(self, value):
        return NeighbourLookupEntryView(self, value)


class NeighbourLookupEntryView(data.View):
    # TODO: remove these, they're unused and outdated.

    # Assuming that this neighbour lookup entry is a T-shaped entry for moving in
    # `t_direction`, return the middle of the arrangement.
    def t_middle(self, t_direction: ValueLike):
        return self.neighbours[opposite(t_direction)]

    # Assuming that this neighbour lookup entry is a T-shaped entry for moving in
    # `t_direction` from `t_input`, return the neighbour in `direction` from the
    # middle of the arrangement.
    def t_neighbour(
        self, t_input: ValueLike, t_direction: ValueLike, direction: ValueLike
    ):
        return Mux(
            direction == opposite(t_direction), t_input, self.neighbours[t_direction]
        )


def neighbour_lookup_layout(max_area: int) -> MemoryData:
    """Returns the layout of one cuboid's neighbour lookup."""

    return MemoryData(
        shape=NeighbourLookupEntryLayout(max_area),
        # It's 5 * max_area elements deep because we need to store:
        #
        # - the regular neighbours of every square
        # - the T-shaped neighbours in all 4 directions from each square
        #
        # thus meaning that for every square, we need to store 5 entries.
        #
        # In terms of the concrete layout, the T-shaped neighbours take up the first 4 *
        # max_area entries, followed by the regular neighbours (so that you can switch
        # to the regular neighbours by just setting the MSB of the address).
        depth=5 * max_area,
        init=[],
    )


class CursorNeighbourLookup(wiring.Component):
    """
    A component for accessing a particular cuboid's neighbour lookup.

    Note that the outputs have a 1-cycle delay on them, as a result of the neighbour
    lookup using a BRAM.
    """

    def __init__(self, max_area: int):
        nl_layout = neighbour_lookup_layout(max_area)

        super().__init__(
            {
                # The port through which the neighbour lookup should be accessed.
                "port": In(
                    ReadPort.Signature(
                        addr_width=ceil_log2(nl_layout.depth), shape=nl_layout.shape
                    )
                ),
                # The cursor to find the neighbours of.
                "input": In(cursor_layout(max_area)),
                # Whether or not to access it in T-shaped mode.
                "t_mode": In(1),
                # If accessing in T-shaped mode, the direction to find the neighbours in.
                "direction": In(2),
                # The middle of the output (i.e., `input` in normal mode and the cursor in
                # `direction` from `input` in T-shaped mode).
                "middle": Out(cursor_layout(max_area)),
                # The neighbours of `middle`.
                "neighbours": Out(cursor_layout(max_area)).array(4),
                # Whether or not `middle` is in the fixed family.
                "fixed_family": Out(1),
                # The transform of `middle`'s class.
                "transform": Out(3),
            }
        )

        self._max_area = max_area

    def elaborate(self, platform) -> Module:
        m = Module()

        # Figure out what direction we need to go in relative to `self.input`'s square.
        square_direction = self.input.orientation + self.direction
        # Then figure our what address that corresponds to.
        addr = Mux(
            self.t_mode,
            square_direction * self._max_area + self.input.square,
            4 * self._max_area + self.input.square,
        )

        m.d.comb += self.port.addr.eq(addr)

        # Because of BRAMs' one-cycle delay, we need a second pipeline stage to process
        # their outputs.
        prev_input = pipe(m, self.input)
        prev_t_mode = pipe(m, self.t_mode)
        prev_direction = pipe(m, self.direction)

        m.d.comb += self.fixed_family.eq(self.port.data.fixed_family)
        m.d.comb += self.transform[2].eq(self.port.data.transform[2])
        m.d.comb += self.transform[:2].eq(
            # NOTE: if we ever want to support less than 2 transform bits, we'll have to
            # truncate here.
            self.port.data.transform[:2] + prev_input.orientation
        )

        # Fix up the orientations of the neighbours we read.
        oriented_neighbours = []
        for i in range(4):
            oriented = Signal(cursor_layout(self._max_area))
            read_data = self.port.data.neighbours[i]

            m.d.comb += oriented.square.eq(read_data.square)
            m.d.comb += oriented.orientation.eq(
                prev_input.orientation + read_data.orientation
            )

            oriented_neighbours.append(oriented)

        m.d.comb += self.middle.eq(Mux(prev_t_mode, oriented_neighbours[0], prev_input))

        # Convert the raw list of possibly-T-shaped neighbours into a list of actual
        # neighbours of the middle cursor, that just needs a bit of rotation.
        unrotated_neighbours = Array(
            [
                Mux(prev_t_mode, prev_input, oriented_neighbours[0]),
                oriented_neighbours[1],
                oriented_neighbours[2],
                oriented_neighbours[3],
            ]
        )

        # The amount to rotate `unrotated_neighbours` right by, putting the result into
        # `self.neighbours`.
        rotation = Signal(2)
        with m.If(prev_t_mode):
            # The middle is what you get by going in `prev_direction` from
            # `unrotated_neighbours[0]`.
            #
            # Thus `unrotated_neighbours[0]` is what you get by going in
            # `opposite(prev_direction)` from the middle.
            #
            # Thus `unrotated_neighbours[0]` should go in
            # `self.neighbours[opposite(prev_direction)]`, and so we need to rotate right by
            # `opposite(prev_direction)`.
            m.d.comb += rotation.eq(opposite(prev_direction))
        with m.Else():
            # `self.neighbours[0]` is in `unrotated_neighbours[prev_input.orientation]`, so
            # we need to rotate left by `prev_input.orientation` to get it into the right
            # spot.
            m.d.comb += rotation.eq(-prev_input.orientation)

        for i in range(4):
            index = (i - rotation)[:2]
            m.d.comb += self.neighbours[i].eq(unrotated_neighbours[index])

        return m


class NeighbourLookup(wiring.Component):
    def __init__(self, cuboids: int, max_area: int):
        nl_layout = neighbour_lookup_layout(max_area)

        super().__init__(
            {
                # The ports through which the neighbour lookups should be accessed.
                "ports": In(
                    ReadPort.Signature(
                        addr_width=ceil_log2(nl_layout.depth), shape=nl_layout.shape
                    )
                ).array(cuboids),
                # The instruction to find neighbours of.
                "input": In(instruction_layout(cuboids, max_area)),
                # Whether or not to access it in T-shaped mode.
                "t_mode": In(1),
                # If accessing in T-shaped mode, the direction to find the neighbours in.
                "direction": In(2),
                # The middle of the output (i.e., `input` in normal mode and the instruction in
                # `direction` from `input` in T-shaped mode).
                "middle": Out(instruction_layout(cuboids, max_area)),
                # The neighbours of `middle`.
                "neighbours": Out(instruction_layout(cuboids, max_area)).array(4),
                # Whether or not the first cursor of `middle` is in the fixed family.
                "fixed_family": Out(1),
                # The transform of the first cursor of `middle`'s class.
                "transform": Out(3),
            }
        )

        self._cuboids = cuboids
        self._max_area = max_area

    def elaborate(self, platform) -> Module:
        m = Module()

        for i in range(self._cuboids):
            lookup = CursorNeighbourLookup(self._max_area)

            wiring.connect(m, lookup.port, wiring.flipped(self.ports[i]))
            m.d.comb += lookup.input.eq(self.input.mapping[i])
            m.d.comb += lookup.t_mode.eq(self.t_mode)
            m.d.comb += lookup.direction.eq(self.direction)

            m.d.comb += self.middle.mapping[i].eq(lookup.middle)
            for j in range(4):
                m.d.comb += self.neighbours[j].mapping[i].eq(lookup.neighbours[j])
            if i == 0:
                m.d.comb += self.fixed_family.eq(lookup.fixed_family)
                m.d.comb += self.transform.eq(lookup.transform)

            m.submodules[f"cursor_neighbour_lookup_{i}"] = lookup

        middle_pos = Mux(
            self.t_mode, self.input.pos.moved_in(m, self.direction), self.input.pos
        )
        middle_pos = PosLayout(self._max_area)(middle_pos)
        neighbour_pos = [
            middle_pos.moved_in(m, Const(direction, 2)) for direction in range(4)
        ]

        # We need to match `CursorNeighbourLookup`'s 1-cycle delay.
        prev_middle_pos = pipe(m, middle_pos)
        prev_neighbour_pos = [pipe(m, neighbour_pos[i]) for i in range(4)]

        m.d.comb += self.middle.pos.eq(prev_middle_pos)
        for i in range(4):
            m.d.comb += self.neighbours[i].pos.eq(prev_neighbour_pos[i])

        return m
