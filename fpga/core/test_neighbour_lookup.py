import json
import subprocess

from amaranth import *
from amaranth.lib import wiring
from amaranth.lib.memory import Memory
from amaranth.sim import Simulator

from core.base_types import cursor_layout
from core.neighbour_lookup import (
    NeighbourLookup,
    NeighbourLookupEntryLayout,
    neighbour_lookup_layout,
)


def test_neighbour_lookup():
    res = subprocess.run(
        ["cargo", "run", "--release", "--bin", "dump_neighbours"],
        stdout=subprocess.PIPE,
    )
    data = json.loads(res.stdout)

    dut = Module()

    neighbour_lookup = NeighbourLookup(2, 64)
    dut.submodules["neighbour_lookup"] = neighbour_lookup

    for i, init in enumerate(data["neighbour_lookup_contents"]):
        memory = Memory(
            neighbour_lookup_layout(
                64,
                init=[
                    NeighbourLookupEntryLayout(64).from_bits(entry) for entry in init
                ],
            )
        )
        wiring.connect(dut, neighbour_lookup.ports[i], memory.read_port())
        dut.submodules[f"neighbour_lookup_mem_{i}"] = memory

    async def testbench(ctx):
        def set_instruction(signal, data):
            ctx.set(signal.pos.x, data["pos"]["x"])
            ctx.set(signal.pos.y, data["pos"]["y"])
            for i in range(2):
                ctx.set(
                    signal.mapping[i], cursor_layout(64).from_bits(data["mapping"][i])
                )

        def assert_instruction_eq(signal, data):
            assert ctx.get(signal.pos.x) == data["pos"]["x"]
            assert ctx.get(signal.pos.y) == data["pos"]["y"]
            for i in range(2):
                assert ctx.get(signal.mapping[i]) == cursor_layout(64).from_bits(
                    data["mapping"][i]
                )

        for test_case in data["test_cases"]:
            set_instruction(neighbour_lookup.input, test_case["input"])
            ctx.set(neighbour_lookup.t_mode, test_case["t_mode"])
            ctx.set(neighbour_lookup.direction, test_case["direction"])

            await ctx.tick()

            assert_instruction_eq(neighbour_lookup.middle, test_case["middle"])
            for i in range(4):
                assert_instruction_eq(
                    neighbour_lookup.neighbours[i], test_case["neighbours"][i]
                )
            assert ctx.get(neighbour_lookup.fixed_family) == test_case["fixed_family"]
            if test_case["fixed_family"]:
                assert ctx.get(neighbour_lookup.transform) == test_case["transform"]

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    sim.run()
