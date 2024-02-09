import os
import subprocess

from core import CoreManager, Cuboid
from litepcie.phy.s7pciephy import S7PCIEPHY
from litepcie.software import generate_litepcie_software
from litex.gen import *
from litex.soc.cores.clock import S7IDELAYCTRL, S7PLL
from litex.soc.cores.dna import DNA
from litex.soc.cores.xadc import XADC
from litex.soc.integration.builder import *
from litex.soc.integration.soc_core import *
from litex_boards.platforms import sqrl_acorn
from litex_boards.targets.sqrl_acorn import CRG


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
        platform.add_false_path_constraints(
            self.cd_sys.clk, pll.clkin
        )  # Ignore sys_clk to pll.clkin path created by SoC's rst.

        self.idelayctrl = S7IDELAYCTRL(self.cd_idelay)


class SoC(SoCCore):
    def __init__(
        self,
        cuboids: list[Cuboid],
        cores: int,
        variant="cle-215",
        sys_clk_freq=66.67e6,
        core_clk_freq=60e6,
        with_analyzer: bool = False,
        **kwargs,
    ):
        for cuboid in cuboids:
            assert cuboid.surface_area() == cuboids[0].surface_area()

        platform = Platform(variant=variant)

        # CRG --------------------------------------------------------------------------------------
        self.crg = CRG(platform, sys_clk_freq, core_clk_freq)

        # SoCCore ----------------------------------------------------------------------------------
        SoCCore.__init__(
            self,
            platform,
            sys_clk_freq,
            ident=f"net-finder LiteX SoC on Acorn CLE-101/215(+) (solving {', '.join(map(str, cuboids))})",
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
        platform.add_period_constraint(self.crg.cd_sys.clk, 1e9 / sys_clk_freq)

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

        # I would use commas here but that breaks CSV export.
        self.add_config("CUBOIDS", ";".join(map(str, cuboids)))
        self.core_mgr = CoreManager(cuboids, cores, with_analyzer=with_analyzer)


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
        "--sys-clk-freq", default=56.25e6, type=float, help="System clock frequency."
    )
    parser.add_target_argument(
        "--core-clk-freq", default=56.25e6, type=float, help="Core clock frequency."
    )
    parser.add_target_argument(
        "--driver", action="store_true", help="Generate PCIe driver."
    )
    parser.add_target_argument(
        "--cuboids", nargs="+", help="The cuboids to find nets of."
    )
    parser.add_target_argument(
        "--cores", default=75, type=int, help="The number of cores to include."
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
        vivado_synth_directive="default -flatten_hierarchy none",
    )

    args = parser.parse_args()

    cuboids = [Cuboid(s) for s in args.cuboids]

    soc = SoC(
        variant=args.variant,
        sys_clk_freq=args.sys_clk_freq,
        core_clk_freq=args.core_clk_freq,
        cuboids=cuboids,
        cores=args.cores,
        with_analyzer=args.with_analyzer,
        **parser.soc_argdict,
    )

    builder = Builder(soc, **parser.builder_argdict)
    if args.build:
        sources_dir = os.path.join(
            os.path.dirname(__file__),
            "net-finder.srcs/sources_1/new",
        )
        subprocess.run(
            [
                "cargo",
                "run",
                "--bin",
                "fpga_gen",
                sources_dir,
                *args.cuboids,
            ]
        )
        soc.platform.add_sources(
            sources_dir,
            "generated.svh",
            "types.svh",
            "generated.sv",
            "instruction_neighbour.sv",
            "valid_checker.sv",
            "core.sv",
            copy=True,
        )
        builder.build(**parser.toolchain_argdict)
    else:
        # This allows generating the LitePCIe driver without performing a build.
        soc.finalize()

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
