import json
import subprocess

import pytest
from amaranth import *
from amaranth.lib import wiring
from amaranth.lib.memory import Memory
from amaranth.sim import Simulator

from .base_types import cursor_layout
from .skip_checker import SkipChecker, undo_lookup_entry_layout, undo_lookup_layout


@pytest.mark.parametrize("cuboids", [["1x1x5", "1x2x3"], ["1x1x7", "1x3x3"]])
def test_neighbour_lookup(cuboids):
    res = subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "dump_skipped",
            "--features",
            "no-trie",
            *cuboids,
        ],
        stdout=subprocess.PIPE,
    )
    data = json.loads(res.stdout)

    dut = Module()

    skip_checker = SkipChecker(2, 64)
    dut.submodules.skip_checker = skip_checker

    for i, init in enumerate(data["undo_lookup_contents"]):
        memory = Memory(
            undo_lookup_layout(
                64,
                init=[undo_lookup_entry_layout(64).from_bits(entry) for entry in init],
            )
        )
        wiring.connect(dut, skip_checker.ports[i], memory.read_port())
        dut.submodules[f"undo_lookup_mem_{i}"] = memory

    async def testbench(ctx):
        for test_case in data["test_cases"]:
            ctx.set(skip_checker.start_mapping_index, test_case["start_mapping_index"])
            for i in range(2):
                ctx.set(
                    skip_checker.input[i],
                    cursor_layout(64).from_bits(test_case["input"][i]),
                )
            ctx.set(skip_checker.fixed_family, test_case["fixed_family"])
            ctx.set(skip_checker.transform, test_case["transform"])

            await ctx.tick()

            assert ctx.get(skip_checker.skip) == test_case["skip"]

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    sim.run()
