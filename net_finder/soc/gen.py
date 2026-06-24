import os
from dataclasses import dataclass
from math import floor
from os import path

from litepcie.phy.s7pciephy import S7PCIEPHY
from litepcie.software import generate_litepcie_software
from litex.gen import *
from litex.soc.cores.clock import S7IDELAYCTRL, S7PLL
from litex.soc.cores.dna import DNA
from litex.soc.cores.xadc import XADC
from litex.soc.integration.builder import *
from litex.soc.integration.soc_core import *
from litex_boards.platforms import sqrl_acorn

from .core import CoreManager

CORE_LUTS = 1477
CORE_BRAMS = 3.5


@dataclass
class PBlock:
    slice_start_x: int
    slice_start_y: int
    slice_end_x: int
    slice_end_y: int
    # In terms of RAMB36s; we multiply the y positions by 2 to get the RAMB18 coordinates.
    bram_start_x: int
    bram_start_y: int
    bram_end_x: int
    bram_end_y: int

    def slices(self):
        return (self.slice_end_x - self.slice_start_x + 1) * (
            self.slice_end_y - self.slice_start_y + 1
        )

    def luts(self):
        return 4 * self.slices()

    def brams(self):
        return (self.bram_end_x - self.bram_start_x + 1) * (
            self.bram_end_y - self.bram_start_y + 1
        )

    def cores(self):
        return min(self.luts() // CORE_LUTS, floor(self.brams() / CORE_BRAMS))


PBLOCKS = [
    PBlock(0, 200, 15, 249, 0, 40, 0, 49),
    PBlock(0, 100, 23, 199, 0, 20, 1, 39),
    # The distribution of SLICEMs is uneven here, divide it up into columns which
    # each have enough to stop cores accidentally starving each other.
    PBlock(0, 0, 31, 99, 0, 0, 1, 19),
    PBlock(32, 0, 51, 99, 2, 0, 2, 19),
    #
    PBlock(52, 50, 113, 99, 3, 10, 6, 19),
    PBlock(52, 150, 113, 199, 3, 30, 6, 39),
    PBlock(114, 0, 163, 249, 7, 0, 8, 49),
]
CORES = sum(pblock.cores() for pblock in PBLOCKS)


class Platform(sqrl_acorn.Platform):
    def __init__(self, variant="cle-215"):
        super().__init__(variant=variant)

        self.toolchain.bitstream_commands += [
            "set_property BITSTREAM.CONFIG.OVERTEMPPOWERDOWN ENABLE [current_design]",
        ]

        for i, cmd in enumerate(self.toolchain.additional_commands):
            if (
                cmd
                == "set_property BITSTREAM.CONFIG.TIMER_CFG 0x0001fbd0 [current_design]"
            ):
                # Extend the configuration timeout from ~0.5s to ~2s; 0.5s doesn't seem to be
                # long enough for a design that fills the whole FPGA.
                self.toolchain.additional_commands[i] = (
                    "set_property BITSTREAM.CONFIG.TIMER_CFG 0x0007a120 [current_design]"
                )


class CRG(LiteXModule):
    def __init__(self, platform, sys_clk_freq, core_clk_freq):
        self.rst = Signal()
        self.cd_sys = ClockDomain()
        self.cd_idelay = ClockDomain()
        self.cd_core = ClockDomain()

        # Clk/Rst
        clk200 = platform.request("clk200")

        # PLL
        self.pll = pll = S7PLL()
        self.comb += pll.reset.eq(self.rst)
        pll.register_clkin(clk200, 200e6)
        pll.create_clkout(self.cd_sys, sys_clk_freq)
        pll.create_clkout(self.cd_core, core_clk_freq)
        pll.create_clkout(self.cd_idelay, 200e6)

        self.idelayctrl = S7IDELAYCTRL(self.cd_idelay)


class SoC(SoCCore):
    def __init__(
        self,
        cuboids: int,
        max_area: int,
        cores: int,
        variant="cle-215",
        sys_clk_freq=50e6,
        core_clk_freq=100e6,
        with_analyzer: bool = False,
        **kwargs,
    ):
        platform = Platform(variant=variant)

        # CRG --------------------------------------------------------------------------------------
        self.crg = CRG(platform, sys_clk_freq, core_clk_freq)

        # SoCCore ----------------------------------------------------------------------------------
        SoCCore.__init__(
            self,
            platform,
            sys_clk_freq,
            ident="net-finder LiteX SoC on Acorn CLE-101/215(+)",
            **kwargs,
        )

        # XADC -------------------------------------------------------------------------------------
        self.xadc = XADC()

        # DNA --------------------------------------------------------------------------------------
        self.dna = DNA()
        self.dna.add_timing_constraints(platform, sys_clk_freq, self.crg.cd_sys.clk)

        # PCIe -------------------------------------------------------------------------------------
        self.comb += platform.request("pcie_clkreq_n").eq(0)
        self.pcie_phy = S7PCIEPHY(
            platform, platform.request("pcie_x4"), data_width=128, bar0_size=0x20000
        )
        self.add_pcie(phy=self.pcie_phy, ndmas=0, address_width=64)

        # ICAP (For FPGA reload over PCIe).
        from litex.soc.cores.icap import ICAP

        self.icap = ICAP()
        self.icap.add_reload()
        self.icap.add_timing_constraints(platform, sys_clk_freq, self.crg.cd_sys.clk)

        # Flash (For SPIFlash update over PCIe).
        from litex.soc.cores.gpio import GPIOOut
        from litex.soc.cores.spi_flash import S7SPIFlash

        self.flash_cs_n = GPIOOut(platform.request("flash_cs_n"))
        self.flash = S7SPIFlash(platform.request("flash"), sys_clk_freq, 25e6)

        self.core_mgr = CoreManager(
            platform, cuboids, max_area, cores, with_analyzer=with_analyzer
        )

        cores_allocated = 0
        for i, pblock in enumerate(PBLOCKS):
            self.platform.toolchain.pre_placement_commands.add(
                f"create_pblock pblock_cores_{i}"
            )

            start_slice = f"SLICE_X{pblock.slice_start_x}Y{pblock.slice_start_y}"
            end_slice = f"SLICE_X{pblock.slice_end_x}Y{pblock.slice_end_y}"
            start_ramb18 = f"RAMB18_X{pblock.bram_start_x}Y{2 * pblock.bram_start_y}"
            end_ramb18 = f"RAMB18_X{pblock.bram_end_x}Y{2 * pblock.bram_end_y + 1}"
            start_ramb36 = f"RAMB36_X{pblock.bram_start_x}Y{pblock.bram_start_y}"
            end_ramb36 = f"RAMB36_X{pblock.bram_end_x}Y{pblock.bram_end_y}"

            # We need 8 curly brackets to survive the processing by the f-strings and the
            # two rounds (sigh) of processing done by LiteX.
            self.platform.toolchain.pre_placement_commands.add(
                f"resize_pblock [get_pblocks pblock_cores_{i}] -add {{{{{{{{{start_slice}:{end_slice}}}}}}}}}"
            )
            self.platform.toolchain.pre_placement_commands.add(
                f"resize_pblock [get_pblocks pblock_cores_{i}] -add {{{{{{{{{start_ramb18}:{end_ramb18}}}}}}}}}"
            )
            self.platform.toolchain.pre_placement_commands.add(
                f"resize_pblock [get_pblocks pblock_cores_{i}] -add {{{{{{{{{start_ramb36}:{end_ramb36}}}}}}}}}"
            )

            cores = pblock.cores()
            for _ in range(cores):
                # TODO: put in_sync / out_sync / synchroniser flip-flops in here too? Right now
                # they seem to be getting placed quite far away
                self.platform.toolchain.pre_placement_commands.add(
                    f"add_cells_to_pblock [get_pblocks pblock_cores_{i}] [get_cells -quiet [list core_group/core_{cores_allocated}]]"
                )
                cores_allocated += 1

        self.platform.toolchain.pre_placement_commands.add("create_pblock pblock_pcie")
        self.platform.toolchain.pre_placement_commands.add(
            "resize_pblock [get_pblocks pblock_pcie] -add {{{{SLICE_X24Y200:SLICE_X51Y249}}}}"
        )
        self.platform.toolchain.pre_placement_commands.add(
            "resize_pblock [get_pblocks pblock_pcie] -add {{{{RAMB18_X1Y80:RAMB18_X2Y99}}}}"
        )
        self.platform.toolchain.pre_placement_commands.add(
            "resize_pblock [get_pblocks pblock_pcie] -add {{{{RAMB36_X1Y40:RAMB36_X2Y49}}}}"
        )
        self.platform.toolchain.pre_placement_commands.add(
            "resize_pblock [get_pblocks pblock_pcie] -add {{{{PCIE_X0Y0:PCIE_X0Y0}}}}"
        )
        self.platform.toolchain.pre_placement_commands.add(
            "resize_pblock [get_pblocks pblock_pcie] -add {{{{GTPE2_COMMON_X0Y1:GTPE2_COMMON_X0Y1}}}}"
        )
        self.platform.toolchain.pre_placement_commands.add(
            "resize_pblock [get_pblocks pblock_pcie] -add {{{{GTPE2_CHANNEL_X0Y4:GTPE2_CHANNEL_X0Y7}}}}"
        )
        self.platform.toolchain.pre_placement_commands.add(
            f"add_cells_to_pblock [get_pblocks pblock_pcie] [get_cells -quiet [list pcie_s7]]"
        )

    def do_finalize(self):
        xdc_path = path.join(self.platform.output_dir, "gateware", "clocks.xdc")
        with open(xdc_path, "w") as f:
            f.write(
                """\
set_clock_groups \\
    -group [get_clocks -include_generated_clocks clk200_p] \\
    -group [get_clocks -include_generated_clocks dna_clk] \\
    -group [get_clocks -include_generated_clocks icap_clk] \\
    -group [get_clocks -include_generated_clocks pcie_x4_clk_p] \\
    -group [get_clocks -include_generated_clocks txoutclk_x0y0] \\
    -asynchronous"""
            )
        self.platform.toolchain.pre_synthesis_commands.add("read_xdc clocks.xdc")
        self.platform.toolchain.pre_synthesis_commands.add(
            "set_property PROCESSING_ORDER LATE [get_files clocks.xdc]"
        )


# Build --------------------------------------------------------------------------------------------


def main():
    from litex.build.parser import LiteXArgumentParser

    parser = LiteXArgumentParser(
        platform=sqrl_acorn.Platform,
        description="net-finder LiteX SoC on Acorn CLE-101/215(+).",
    )

    parser.add_target_argument("--flash", action="store_true", help="Flash bitstream.")
    parser.add_target_argument(
        "--variant",
        default="cle-215",
        help="Board variant (cle-215+, cle-215 or cle-101).",
    )
    parser.add_target_argument(
        "--sys-clk-freq", default=50e6, type=float, help="System clock frequency."
    )
    parser.add_target_argument(
        "--core-clk-freq", default=100e6, type=float, help="Core clock frequency."
    )
    parser.add_target_argument(
        "--driver", action="store_true", help="Generate PCIe driver."
    )
    parser.add_target_argument(
        "--cuboids",
        default=3,
        type=int,
        help="The number of cuboids to find common nets of.",
    )
    parser.add_target_argument(
        "--max-area",
        default=64,
        type=int,
        help="The maximum area of cuboids to find common nets of.",
    )
    parser.add_argument(
        "--with-analyzer", action="store_true", help="Enable Analyzer support."
    )

    # Disable all the features we aren't using (by default).
    parser.set_defaults(
        cpu_type="None",
        integrated_sram_size=0,
        no_uart=True,
        no_timer=True,
        soc_json="soc_info.json",
        soc_csv="soc_info.csv",
        # Somewhat counter-intuitively, disabling flattening seems to actually improve
        # timing (as well as making debugging timing much easier).
        #
        # Also this is an awful hacky way of doing it but I don't see an alternative.
        vivado_synth_directive="default -flatten_hierarchy none -global_retiming on",
    )

    args = parser.parse_args()

    soc = SoC(
        variant=args.variant,
        sys_clk_freq=args.sys_clk_freq,
        core_clk_freq=args.core_clk_freq,
        cuboids=args.cuboids,
        max_area=args.max_area,
        cores=CORES,
        with_analyzer=args.with_analyzer,
        **parser.soc_argdict,
    )

    builder = Builder(soc, **parser.builder_argdict)
    if args.build:
        builder.build(**parser.toolchain_argdict)

    if args.driver:
        generate_litepcie_software(soc, os.path.join(builder.output_dir, "driver"))

    if args.load:
        prog = soc.platform.create_programmer("vivado")
        prog.load_bitstream(builder.get_bitstream_filename(mode="sram"))

    if args.flash:
        prog = soc.platform.create_programmer("vivado")
        prog.flash(0, builder.get_bitstream_filename(mode="flash"))


if __name__ == "__main__":
    main()
