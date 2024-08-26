import enum

from amaranth import *
from amaranth.lib import wiring
from amaranth.lib.memory import ReadPort
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from net_finder.core.neighbour_lookup import NeighbourLookup
from net_finder.core.net import Net, shard_depth

from .base_types import instruction_layout, net_size
from .core import FINDERS_PER_CORE, run_stack_entry_layout
from .memory import ChunkedMemory
from .neighbour_lookup import neighbour_lookup_layout
from .skip_checker import SkipChecker, undo_lookup_layout
from .utils import pipe


def child_index_to_direction(children: int, child_index: int) -> int | None:
    """
    Returns the direction of the `child_index`'th valid child, where `children` is a
    bitfield of whether the child in each direction is valid.

    In other words, returns the index in `children` of the `child_index`'th set bit.

    If `child_index` is >= the number of set bits in `children`, it's interpreted as
    meaning the direction of the last valid child.

    Returns None if there are no bits set in `children`.
    """

    current_index = -1
    last_valid_direction = None
    for direction in range(4):
        if children & (1 << direction):
            current_index += 1
            last_valid_direction = direction
            if current_index == child_index:
                return direction

    return last_valid_direction


class Task(enum.Enum):
    """The different tasks that a `MainPipeline` can perform."""

    Clear = 0
    """
    We need to clear an address in this finder's net/surfaces.

    The potential surfaces clear themselves in the background, and everything else
    has some kind of length associated with it which makes it fine for things beyond
    that length to be garbage.
    """
    Advance = 1
    """
    We need to check whether the input instruction is valid, and run it if so.

    'valid' means that none of the squares the instruction tries to set on the
    cuboids' surfaces are already filled, it isn't skipped, and at least one of its
    neighbours is valid.

    The definition for a neighbour being valid is similar, except that we don't
    check for whether they're skipped yet and they're considered invalid if there's
    already another instruction in the queue that sets the same net position.
    """
    Check = 2
    """
    We need to check whether the input instruction and its neighbours are valid.

    This is used both for CHECK state, and for RECEIVE state if the next decision is
    0 (we need to know whether or not the instruction is valid to know if that
    decision's been consumed). It's also used as a 'do nothing' task.
    """
    Backtrack = 3
    """
    We need to backtrack the input instruction.
    """


class MainPipeline(wiring.Component):
    """The main pipeline of a `Core`."""

    def __init__(self, cuboids: int, max_area: int):
        self._cuboids = cuboids
        self._max_area = max_area

        nl_layout = neighbour_lookup_layout(max_area)
        ul_layout = undo_lookup_layout(max_area)

        super().__init__(
            {
                # Which finder's state we should be operating on.
                "finder": In(range(FINDERS_PER_CORE)),
                # The index of `finder`'s start mapping.
                "start_mapping_index": In((cuboids - 1) * ceil_log2(max_area)),
                # The task we need to perform.
                "task": In(Task),
                # The run stack entry we're operating on.
                "entry": In(run_stack_entry_layout(cuboids, max_area)),
                # The index of the child of `self.entry` we're operating on (if we're advancing or
                # checking).
                "child_index": In(2),
                # The address in the net/surfaces that we need to clear.
                "clear_index": In(range(shard_depth(max_area))),
                # The ports this pipeline should use to access the neighbour lookups.
                "neighbour_lookups": In(
                    ReadPort.Signature(
                        addr_width=ceil_log2(nl_layout.depth),
                        shape=nl_layout.shape,
                    )
                ).array(cuboids),
                # The ports this pipeline should use to access the undo lookups.
                "undo_lookups": In(
                    ReadPort.Signature(
                        addr_width=ceil_log2(ul_layout.depth),
                        shape=ul_layout.shape,
                    )
                ).array(cuboids - 1),
                # The instruction the pipeline ended up operating on - so, the neighbour of
                # `entry` when advancing/checking, and `entry.instruction` itself when
                # backtracking.
                "instruction": Out(instruction_layout(cuboids, max_area)),
                # Whether or not `instruction` was valid.
                "instruction_valid": Out(1),
                # Whether or not the neighbours of `instruction` in each direction were valid.
                "neighbours_valid": Out(4),
            }
        )

    def elaborate(self, platform) -> Module:
        m = Module()

        net = Net(self._max_area, FINDERS_PER_CORE)
        m.submodules.net = net

        middle_read_ports = []
        middle_write_ports = []
        neighbour_ports = []
        for i in range(self._cuboids):
            surface = ChunkedMemory(
                shape=1, depth=self._max_area, chunks=FINDERS_PER_CORE
            )
            m.submodules[f"surface{i}"] = surface

            read_port, write_port = surface.sdp_port()
            middle_read_ports.append(read_port)
            middle_write_ports.append(write_port)
            neighbour_ports.append([surface.read_port() for i in range(4)])

        # Neighbour lookup (NL) stage
        #
        # This is the stage where we calculate the addresses to index into the neighbour
        # lookup RAMs with.
        nl_finder = self.finder
        nl_start_mapping_index = self.start_mapping_index
        nl_task = self.task
        nl_entry = self.entry
        nl_child_index = self.child_index
        nl_clear_index = self.clear_index

        # Figure out which direction the child at index `child_index` is in.
        nl_child_direction = Signal(2)
        with m.Switch(Cat(nl_entry.children, nl_child_index)):
            for value in range(64):
                children = value & 0b001111
                child_index = value & 0b110000
                direction = child_index_to_direction(children, child_index)

                if direction is not None:
                    with m.Case(value):
                        m.d.comb += nl_child_direction.eq(direction)

                # The scenario where `direction` is None (no valid children) should never happen
                # (since we would have marked it as a potential instruction); let Amaranth leave
                # `child_direction` as its default 0.

        neighbour_lookup = NeighbourLookup(self._cuboids, self._max_area)
        m.submodules.neighbour_lookup = neighbour_lookup

        for i in range(self._cuboids):
            wiring.connect(
                m,
                neighbour_lookup.ports[i],
                wiring.flipped(self.neighbour_lookups[i]),
            )
        m.d.comb += neighbour_lookup.input.eq(nl_entry.instruction)
        m.d.comb += neighbour_lookup.t_mode.eq(nl_task != Task.Backtrack)
        m.d.comb += neighbour_lookup.direction.eq(nl_child_direction)

        # Valid check (VC) stage
        #
        # This is the stage where we calculate the addresses to index into the net,
        # surfaces and undo lookup RAMs with.
        vc_finder = pipe(m, nl_finder)
        vc_start_mapping_index = pipe(m, nl_start_mapping_index)
        vc_task = pipe(m, nl_task)
        vc_entry = pipe(m, nl_entry)
        vc_clear_index = pipe(m, nl_clear_index)

        vc_middle = neighbour_lookup.middle
        vc_neighbours = neighbour_lookup.neighbours
        vc_fixed_family = neighbour_lookup.fixed_family
        vc_transform = neighbour_lookup.transform

        m.d.comb += net.read_port.chunk.eq(vc_finder)
        m.d.comb += net.read_port.pos.eq(vc_middle.pos)
        for i in range(4):
            m.d.comb += net.read_port.neighbours[i].eq(vc_neighbours[i].pos)

        for cuboid in range(self._cuboids):
            m.d.comb += middle_read_ports[cuboid].chunk.eq(vc_finder)
            m.d.comb += middle_read_ports[cuboid].addr.eq(
                vc_middle.mapping[cuboid].square
            )

            for dir in range(4):
                m.d.comb += neighbour_ports[cuboid][dir].chunk.eq(vc_finder)
                m.d.comb += neighbour_ports[cuboid][dir].addr.eq(
                    vc_neighbours[dir].mapping[cuboid].square
                )

        skip_checker = SkipChecker(self._cuboids, self._max_area)
        m.submodules.skip_checker = skip_checker

        for i in range(self._cuboids - 1):
            wiring.connect(
                m,
                skip_checker.ports[i],
                wiring.flipped(self.undo_lookups[i]),
            )
        m.d.comb += skip_checker.start_mapping_index.eq(vc_start_mapping_index)
        m.d.comb += skip_checker.input.eq(vc_middle.mapping)
        m.d.comb += skip_checker.fixed_family.eq(vc_fixed_family)
        m.d.comb += skip_checker.transform.eq(vc_transform)

        # Write back (WB) stage
        #
        # This is the stage where we write back any changes that were made to the net
        # and surfaces.
        #
        # This occurs at the same time as the outer pipeline's IF stage.

        wb_finder = pipe(m, vc_finder)
        wb_task = pipe(m, vc_task)
        wb_entry = pipe(m, vc_entry)
        wb_clear_index = pipe(m, vc_clear_index)
        wb_middle = pipe(m, vc_middle)
        wb_neighbours = [pipe(m, neighbour) for neighbour in vc_neighbours]

        wb_middle_filled = Cat(port.data for port in middle_read_ports).any()
        wb_middle_skipped = skip_checker.skip

        wb_neighbours_queued = net.read_port.data
        wb_neighbours_filled = [
            Cat(
                neighbour_ports[cuboid][dir].data for cuboid in range(self._cuboids)
            ).any()
            for dir in range(4)
        ]

        wb_neighbours_valid = [
            ~queued & ~filled
            for queued, filled in zip(wb_neighbours_queued, wb_neighbours_filled)
        ]
        wb_middle_valid = ~wb_middle_filled & ~wb_middle_skipped
        # Whether we should run `middle` (if the task is `Task.Advance`).
        wb_run = wb_middle_valid & Cat(*wb_neighbours_valid).any()

        m.d.comb += net.write_port.chunk.eq(wb_finder)
        m.d.comb += net.write_port.pos.eq(wb_middle.pos)
        for i in range(4):
            with m.If(wb_task == Task.Clear):
                coord_bits = ceil_log2(net_size(self._max_area))
                m.d.comb += net.write_port.neighbours[i].x.eq(
                    wb_clear_index[:coord_bits]
                )
                m.d.comb += net.write_port.neighbours[i].y.eq(
                    wb_clear_index[coord_bits:]
                )
            with m.Else():
                m.d.comb += net.write_port.neighbours[i].eq(wb_neighbours[i].pos)

            m.d.comb += net.write_port.data[i].eq(wb_task == Task.Advance)
            m.d.comb += net.write_port.en[i].eq(
                (wb_task == Task.Clear)
                | ((wb_task == Task.Advance) & wb_run & wb_neighbours_valid[i])
                | ((wb_task == Task.Backtrack) & wb_entry.children[i])
            )

        for i in range(self._cuboids):
            m.d.comb += middle_write_ports[i].chunk.eq(wb_finder)
            m.d.comb += middle_write_ports[i].addr.eq(
                Mux(
                    wb_task == Task.Clear,
                    wb_clear_index,
                    wb_middle.mapping[i].square,
                )
            )
            m.d.comb += middle_write_ports[i].data.eq(wb_task == Task.Advance)
            m.d.comb += middle_write_ports[i].en.eq(
                ((wb_task == Task.Clear) & (wb_clear_index < self._max_area))
                | ((wb_task == Task.Advance) & wb_run)
                | (wb_task == Task.Backtrack)
            )

        m.d.comb += self.instruction_valid.eq(wb_middle_valid)
        m.d.comb += self.neighbours_valid.eq(Cat(*wb_neighbours_valid))

        return m
