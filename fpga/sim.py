import os
import subprocess

from core import CoreManager, Cuboid
from liteeth.mac import LiteEthMAC
from liteeth.phy.gmii import LiteEthPHYGMII
from liteeth.phy.model import LiteEthPHYModel
from liteeth.phy.xgmii import LiteEthPHYXGMII
from litex.build.generic_platform import *
from litex.build.io import DifferentialInput
from litex.build.sim import SimPlatform
from litex.build.sim.config import SimConfig
from litex.gen import *
from litex.soc.integration.builder import *
from litex.soc.integration.soc_core import *
from litex.tools.litex_sim import generate_gtkw_savefile
from migen.genlib.resetsync import AsyncResetSynchronizer

# IOs ----------------------------------------------------------------------------------------------

_io = [
    # Clk / Rst.
    ("sys_clk", 0, Pins(1)),
    ("core_clk", 0, Pins(1)),
    # Ethernet (Stream Endpoint).
    (
        "eth_clocks",
        0,
        Subsignal("tx", Pins(1)),
        Subsignal("rx", Pins(1)),
    ),
    (
        "eth",
        0,
        Subsignal("source_valid", Pins(1)),
        Subsignal("source_ready", Pins(1)),
        Subsignal("source_data", Pins(8)),
        Subsignal("sink_valid", Pins(1)),
        Subsignal("sink_ready", Pins(1)),
        Subsignal("sink_data", Pins(8)),
    ),
    # Ethernet (XGMII).
    (
        "xgmii_eth",
        0,
        Subsignal("rx_data", Pins(64)),
        Subsignal("rx_ctl", Pins(8)),
        Subsignal("tx_data", Pins(64)),
        Subsignal("tx_ctl", Pins(8)),
    ),
    # Ethernet (GMII).
    (
        "gmii_eth",
        0,
        Subsignal("rx_data", Pins(8)),
        Subsignal("rx_dv", Pins(1)),
        Subsignal("rx_er", Pins(1)),
        Subsignal("tx_data", Pins(8)),
        Subsignal("tx_en", Pins(1)),
        Subsignal("tx_er", Pins(1)),
    ),
]

# Platform -----------------------------------------------------------------------------------------


class Platform(SimPlatform):
    def __init__(self):
        SimPlatform.__init__(self, "SIM", _io)


# CRG ---------------------------------------------------------------------------------------------


# The version in `litex.build.generic_platform` doesn't support resets properly
# so I had to tweak it (`self.rst` has to exist for it to be picked up and set
# by `ctrl_reset`).
class CRG(Module):
    def __init__(self, sys_clk, core_clk):
        self.rst = Signal()

        self.clock_domains.cd_sys = ClockDomain()
        self.clock_domains.cd_core = ClockDomain()
        self.clock_domains.cd_por = ClockDomain(reset_less=True)

        if hasattr(sys_clk, "p"):
            sys_clk_se = Signal()
            self.specials += DifferentialInput(sys_clk.p, sys_clk.n, sys_clk_se)
            sys_clk = sys_clk_se

        if hasattr(core_clk, "p"):
            core_clk_se = Signal()
            self.specials += DifferentialInput(core_clk.p, core_clk.n, core_clk_se)
            core_clk = core_clk_se

        # Power on Reset (vendor agnostic)
        int_rst = Signal(reset=1)
        self.sync.por += int_rst.eq(self.rst)
        self.comb += [
            self.cd_sys.clk.eq(sys_clk),
            # Use the slowest clock for this one so that the reset signal isn't missed.
            self.cd_por.clk.eq(core_clk),
            self.cd_core.clk.eq(core_clk),
            self.cd_core.rst.eq(int_rst),
        ]

        self.specials += AsyncResetSynchronizer(self.cd_sys, int_rst)


# BaseSoC -----------------------------------------------------------------------------------------


class BaseSoC(SoCCore):
    def __init__(
        self,
        cuboids: list[Cuboid],
        cores: int,
        with_ethernet=False,
        ethernet_phy_model="sim",
        with_etherbone=False,
        etherbone_mac_address=0x10E2D5000001,
        etherbone_ip_address="192.168.1.51",
        with_analyzer=False,
        sim_debug=False,
        trace_reset_on=False,
        **kwargs
    ):
        platform = Platform()
        sys_clk_freq = int(1e6)

        # CRG --------------------------------------------------------------------------------------
        self.crg = CRG(
            sys_clk=platform.request("sys_clk"), core_clk=platform.request("core_clk")
        )

        # SoCCore ----------------------------------------------------------------------------------
        SoCCore.__init__(
            self, platform, clk_freq=sys_clk_freq, ident="LiteX Simulation", **kwargs
        )

        # Ethernet / Etherbone PHY -----------------------------------------------------------------
        if with_ethernet or with_etherbone:
            if ethernet_phy_model == "sim":
                self.ethphy = LiteEthPHYModel(self.platform.request("eth", 0))
            elif ethernet_phy_model == "xgmii":
                self.ethphy = LiteEthPHYXGMII(
                    None, self.platform.request("xgmii_eth", 0), model=True
                )
            elif ethernet_phy_model == "gmii":
                self.ethphy = LiteEthPHYGMII(
                    None, self.platform.request("gmii_eth", 0), model=True
                )
            else:
                raise ValueError("Unknown Ethernet PHY model:", ethernet_phy_model)

        # Etherbone with optional Ethernet ---------------------------------------------------------
        if with_etherbone:
            self.add_etherbone(
                phy=self.ethphy,
                ip_address=etherbone_ip_address,
                mac_address=etherbone_mac_address,
                data_width=8,
                with_ethmac=with_ethernet,
            )
        # Ethernet only ----------------------------------------------------------------------------
        elif with_ethernet:
            # Ethernet MAC
            self.ethmac = ethmac = LiteEthMAC(
                phy=self.ethphy,
                dw=64 if ethernet_phy_model == "xgmii" else 32,
                interface="wishbone",
                endianness=self.cpu.endianness,
            )
            ethmac_region_size = (
                ethmac.rx_slots.constant + ethmac.tx_slots.constant
            ) * ethmac.slot_size.constant
            ethmac_region = SoCRegion(
                origin=self.mem_map.get("ethmac", None),
                size=ethmac_region_size,
                cached=False,
            )
            self.bus.add_slave(name="ethmac", slave=ethmac.bus, region=ethmac_region)

            # Add IRQs (if enabled).
            if self.irq.enabled:
                self.irq.add("ethmac", use_loc_if_exists=True)

        # Simulation debugging ----------------------------------------------------------------------
        if sim_debug:
            platform.add_debug(self, reset=1 if trace_reset_on else 0)
        else:
            self.comb += platform.trace.eq(1)

        for cuboid in cuboids:
            assert cuboid.surface_area() == cuboids[0].surface_area()
        # I would use commas here but that breaks CSV export.
        self.add_config("CUBOIDS", ";".join(map(str, cuboids)))
        self.core_mgr = CoreManager(cuboids, cores, with_analyzer=with_analyzer)


# Build --------------------------------------------------------------------------------------------


def main():
    from litex.build.parser import LiteXArgumentParser

    parser = LiteXArgumentParser(description="LiteX SoC Simulation utility")
    parser.set_platform(SimPlatform)

    # Ethernet /Etherbone.
    parser.add_argument(
        "--with-ethernet", action="store_true", help="Enable Ethernet support."
    )
    parser.add_argument(
        "--ethernet-phy-model",
        default="sim",
        help="Ethernet PHY to simulate (sim, xgmii or gmii).",
    )
    parser.add_argument(
        "--with-etherbone", action="store_true", help="Enable Etherbone support."
    )
    parser.add_argument(
        "--local-ip", default="192.168.1.50", help="Local IP address of SoC."
    )
    parser.add_argument(
        "--remote-ip", default="192.168.1.100", help="Remote IP address of TFTP server."
    )

    # Analyzer.
    parser.add_argument(
        "--with-analyzer", action="store_true", help="Enable Analyzer support."
    )

    # Debug/Waveform.
    parser.add_argument(
        "--sim-debug", action="store_true", help="Add simulation debugging modules."
    )
    parser.add_argument(
        "--gtkwave-savefile", action="store_true", help="Generate GTKWave savefile."
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run simulation without user input.",
    )

    parser.add_target_argument(
        "--cuboids", nargs="+", help="The cuboids to find nets of."
    )
    parser.add_target_argument(
        "--cores",
        # Default to just 3 cores in simulation, I don't think the simulator would be
        # very happy simulating 80 of them.
        default=3,
        type=int,
        help="The number of cores to include.",
    )

    # Disable all the features we aren't using (by default).
    parser.set_defaults(
        cpu_type="None",
        integrated_sram_size=0,
        no_uart=True,
        no_timer=True,
        soc_json="sim_soc_info.json",
        soc_csv="sim_soc_info.csv",
    )

    args = parser.parse_args()

    cuboids = [Cuboid(s) for s in args.cuboids]

    soc_kwargs = soc_core_argdict(args)

    sys_clk_freq = int(1e6)
    sim_config = SimConfig()
    sim_config.add_clocker("sys_clk", freq_hz=sys_clk_freq)
    sim_config.add_clocker("core_clk", freq_hz=0.8 * sys_clk_freq)

    # Configuration --------------------------------------------------------------------------------

    # Ethernet.
    if args.with_ethernet or args.with_etherbone:
        if args.ethernet_phy_model == "sim":
            sim_config.add_module(
                "ethernet", "eth", args={"interface": "tap0", "ip": args.remote_ip}
            )
        elif args.ethernet_phy_model == "xgmii":
            sim_config.add_module(
                "xgmii_ethernet",
                "xgmii_eth",
                args={"interface": "tap0", "ip": args.remote_ip},
            )
        elif args.ethernet_phy_model == "gmii":
            sim_config.add_module(
                "gmii_ethernet",
                "gmii_eth",
                args={"interface": "tap0", "ip": args.remote_ip},
            )
        else:
            raise ValueError("Unknown Ethernet PHY model: " + args.ethernet_phy_model)

    # SoC ------------------------------------------------------------------------------------------
    soc = BaseSoC(
        with_ethernet=args.with_ethernet,
        ethernet_phy_model=args.ethernet_phy_model,
        with_etherbone=args.with_etherbone,
        with_analyzer=args.with_analyzer,
        sim_debug=args.sim_debug,
        trace_reset_on=int(float(args.trace_start)) > 0
        or int(float(args.trace_end)) > 0,
        cuboids=cuboids,
        cores=args.cores,
        **soc_kwargs,
    )

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

    if args.with_ethernet:
        for i in range(4):
            soc.add_constant(
                "LOCALIP{}".format(i + 1), int(args.local_ip.split(".")[i])
            )
        for i in range(4):
            soc.add_constant(
                "REMOTEIP{}".format(i + 1), int(args.remote_ip.split(".")[i])
            )

    # Build/Run ------------------------------------------------------------------------------------
    def pre_run_callback(vns):
        if args.trace:
            generate_gtkw_savefile(builder, vns, args.trace_fst)

    builder = Builder(soc, **parser.builder_argdict)
    builder.build(
        sim_config=sim_config,
        interactive=not args.non_interactive,
        pre_run_callback=pre_run_callback,
        **parser.toolchain_argdict,
    )


if __name__ == "__main__":
    main()
