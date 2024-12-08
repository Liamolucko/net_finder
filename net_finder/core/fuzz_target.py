from amaranth import *
from amaranth.hdl import ValueCastable
from amaranth.hdl._ir import PortDirection  # TODO: is there some way to avoid this?
from amaranth.lib import wiring
from amaranth.lib.memory import WritePort
from amaranth.lib.wiring import Out
from amaranth.utils import ceil_log2

from net_finder.core.memory import ConfigMemory

from .core import Core, CoreInterface, State
from .main_pipeline import FINDERS_PER_CORE
from .neighbour_lookup import neighbour_lookup_layout
from .skip_checker import undo_lookup_layout


class FuzzTarget(wiring.Component):
    def __init__(self, cuboids: int, max_area: int):
        nl_layout = neighbour_lookup_layout(max_area)
        ul_layout = undo_lookup_layout(max_area)

        super().__init__(
            {
                "interfaces": Out(CoreInterface(max_area)).array(FINDERS_PER_CORE),
                "neighbour_lookups": Out(
                    WritePort.Signature(
                        addr_width=ceil_log2(nl_layout.depth),
                        shape=nl_layout.shape,
                    )
                ).array(cuboids),
                "undo_lookups": Out(
                    WritePort.Signature(
                        addr_width=ceil_log2(ul_layout.depth),
                        shape=ul_layout.shape,
                    )
                ).array(cuboids - 1),
                "state": Out(State),
            }
        )

        self._cuboids = cuboids
        self._max_area = max_area

    def elaborate(self, platform) -> Module:
        m = Module()

        core = Core(cuboids=self._cuboids, max_area=self._max_area)
        m.submodules.core = core

        for a, b in zip(core.interfaces, self.interfaces):
            wiring.connect(m, a, wiring.flipped(b))

        for i in range(self._cuboids):
            neighbour_lookup = ConfigMemory(
                data=neighbour_lookup_layout(self._max_area)
            )
            m.submodules[f"neighbour_lookup_{i}"] = neighbour_lookup
            wiring.connect(
                m,
                wiring.flipped(self.neighbour_lookups[i]),
                neighbour_lookup.write_port,
            )
            wiring.connect(m, core.neighbour_lookups[i], neighbour_lookup.read_port())

        for i in range(self._cuboids - 1):
            undo_lookup = ConfigMemory(data=undo_lookup_layout(self._max_area))
            m.submodules[f"undo_lookup_{i}"] = undo_lookup
            wiring.connect(
                m, wiring.flipped(self.undo_lookups[i]), undo_lookup.write_port
            )
            wiring.connect(m, core.undo_lookups[i], undo_lookup.read_port())

        return m


if __name__ == "__main__":
    from amaranth.back import verilog

    fuzz_target = FuzzTarget(3, 64)

    # Copied from verilog.convert, except that single underscores are used rather
    # than double ones because Verilator mangles double underscores into oblivion
    # for some reason.
    ports = {}
    for path, member, value in fuzz_target.signature.flatten(fuzz_target):
        if isinstance(value, ValueCastable):
            value = value.as_value()
        if isinstance(value, Value):
            if member.flow == wiring.In:
                dir = PortDirection.Input
            else:
                dir = PortDirection.Output
            ports["_".join(map(str, path))] = (value, dir)

    print(verilog.convert(fuzz_target, ports=ports))
