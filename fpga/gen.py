import os
import subprocess

from litex.gen import *

from litex_boards.platforms import sqrl_acorn
from litex_boards.targets.sqrl_acorn import BaseSoC

from litex.soc.integration.builder import *

from litepcie.software import generate_litepcie_software

from core import CoreManager, Cuboid


class SoC(BaseSoC):
    def __init__(self, cuboids: list[Cuboid], with_analyzer: bool = False, **kwargs):
        super().__init__(with_led_chaser=False, with_pcie=True, **kwargs)

        # Add an extra clock domain for `core`s to run in.
        self.crg.cd_core = ClockDomain()
        # 80MHz is the maximum clock frequency `core` can currently run at.
        self.crg.pll.create_clkout(self.crg.cd_core, 80e6)

        for cuboid in cuboids:
            assert cuboid.surface_area() == cuboids[0].surface_area()
        # I would use commas here but that breaks CSV export.
        self.add_config("CUBOIDS", ";".join(map(str, cuboids)))
        self.core_mgr = CoreManager(cuboids, with_analyzer=with_analyzer)


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
        "--sys-clk-freq", default=100e6, type=float, help="System clock frequency."
    )
    parser.add_target_argument(
        "--driver", action="store_true", help="Generate PCIe driver."
    )
    parser.add_target_argument(
        "--cuboids", nargs="+", help="The cuboids to find nets of."
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
    )

    args = parser.parse_args()

    cuboids = [Cuboid(s) for s in args.cuboids]

    soc = SoC(
        variant=args.variant,
        sys_clk_freq=args.sys_clk_freq,
        cuboids=cuboids,
        with_analyzer=args.with_analyzer,
        **parser.soc_argdict
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
                os.path.join(sources_dir, "generated.sv"),
                *args.cuboids,
            ]
        )
        soc.platform.add_source(os.path.join(sources_dir, "core.sv"))
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
