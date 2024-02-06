use std::fs;
use std::path::PathBuf;

use anyhow::bail;
use clap::Parser;
use litex_bridge::{SocConstant, SocInfo};
use net_finder::Cuboid;
use net_finder_fpga_driver::FpgaRuntime;
use wishbone_bridge::{EthernetBridge, PCIeBridge};

#[derive(Parser)]
#[command(group = clap::ArgGroup::new("bridge").required(true).multiple(false))]
struct Args {
    #[arg(long)]
    resume: bool,
    /// The path of the `soc_info.json` file for the SoC we're driving.
    soc_info: PathBuf,
    /// If the SoC is connected via. PCIe, the path to its folder in
    /// `/sys/bus/pci/devices`.
    #[arg(long, group = "bridge")]
    pcie_device: Option<PathBuf>,
    /// If the SoC is connected via. Etherbone, its IP address.
    #[arg(long, group = "bridge")]
    udp_ip: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let soc_info_json = fs::read_to_string(args.soc_info)?;
    let soc_info: SocInfo = serde_json::from_str(&soc_info_json)?;

    let Some(Some(SocConstant::String(cuboids))) = soc_info.constants.get("config_cuboids") else {
        bail!("unable to read cuboids")
    };

    let cuboids: Vec<Cuboid> = cuboids
        .split(';')
        .map(|s| s.parse::<Cuboid>())
        .collect::<Result<_, _>>()?;

    let bridge = if let Some(ref device) = args.pcie_device {
        PCIeBridge::new(device.join("resource0"))?.create()?
    } else if let Some(ip) = args.udp_ip {
        EthernetBridge::new(ip)?.create()?
    } else {
        unreachable!()
    };
    // Note: When using PCIe, this needs to go right after creating the bridge due
    // to a race condition in `wishbone-bridge`.
    bridge.connect()?;

    let csr_only = args.pcie_device.is_some();

    match *cuboids.as_slice() {
        [] => bail!("must specify at least 1 cuboid"),
        [a] => net_finder::drive(
            FpgaRuntime {
                soc_info,
                cuboids: [a],
                bridge,
                csr_only,
            },
            args.resume,
        ),
        [a, b] => net_finder::drive(
            FpgaRuntime {
                soc_info,
                cuboids: [a, b],
                bridge,
                csr_only,
            },
            args.resume,
        ),
        [a, b, c] => net_finder::drive(
            FpgaRuntime {
                soc_info,
                cuboids: [a, b, c],
                bridge,
                csr_only,
            },
            args.resume,
        ),
        _ => bail!("only up to 3 cuboids are currently supported"),
    }
}
