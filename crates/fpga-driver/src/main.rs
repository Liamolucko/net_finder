use std::fs;
use std::net::TcpStream;
use std::path::PathBuf;

use anyhow::bail;
use clap::Parser;
use litex_bridge::{Bridge, CsrGroup, PcieBridge, SocInfo, UartBridge};
use net_finder::Cuboid;
use net_finder_fpga_driver::{CoreManagerRegisters, FpgaRuntime};

#[derive(Parser)]
#[command(group = clap::ArgGroup::new("bridge").required(true).multiple(false))]
struct Args {
    #[arg(long)]
    resume: bool,
    /// The path of the `soc_info.json` file for the SoC we're driving.
    soc_info: PathBuf,
    cuboids: Vec<Cuboid>,
    /// If the SoC is connected via. PCIe, the path to its folder in
    /// `/sys/bus/pci/devices`.
    #[arg(long, group = "bridge")]
    pcie_device: Option<PathBuf>,
    /// If the SoC is connected via. serial2tcp, its address.
    #[arg(long, group = "bridge")]
    addr: Option<String>,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    let soc_info_json = fs::read_to_string(args.soc_info)?;
    let soc_info: SocInfo = serde_json::from_str(&soc_info_json)?;

    let mut pcie_bridge;
    let mut uart_bridge;
    let bridge: &mut (dyn Bridge + Send) = if let Some(ref device) = args.pcie_device {
        pcie_bridge = PcieBridge::new(device.join("resource0"))?;
        &mut pcie_bridge
    } else if let Some(ip) = args.addr {
        let socket = TcpStream::connect(ip)?;
        uart_bridge = UartBridge::new(socket);
        &mut uart_bridge
    } else {
        unreachable!()
    };

    let csr_only = args.pcie_device.is_some();

    match *args.cuboids.as_slice() {
        [] => bail!("must specify at least 1 cuboid"),
        [a] => net_finder::drive(
            FpgaRuntime {
                soc_info: soc_info.clone(),
                cuboids: [a],
                bridge,
                csr_only,
            },
            args.resume,
        )?,
        [a, b] => net_finder::drive(
            FpgaRuntime {
                soc_info: soc_info.clone(),
                cuboids: [a, b],
                bridge,
                csr_only,
            },
            args.resume,
        )?,
        [a, b, c] => net_finder::drive(
            FpgaRuntime {
                soc_info: soc_info.clone(),
                cuboids: [a, b, c],
                bridge,
                csr_only,
            },
            args.resume,
        )?,
        _ => bail!("only up to 3 cuboids are currently supported"),
    }

    let reg_addrs = CoreManagerRegisters::addrs(&soc_info, csr_only, "core_mgr")?;
    let mut regs = CoreManagerRegisters::backed_by(bridge, reg_addrs);

    fn to_u64(x: [u32; 2]) -> u64 {
        ((x[0] as u64) << 32) | x[1] as u64
    }

    let clear_count = to_u64(regs.clear_count().read()?);
    let receive_count = to_u64(regs.receive_count().read()?);
    let run_count = to_u64(regs.run_count().read()?);
    let check_count = to_u64(regs.check_count().read()?);
    let solution_count = to_u64(regs.solution_count().read()?);
    let split_count = to_u64(regs.split_count().read()?);
    let pause_count = to_u64(regs.pause_count().read()?);

    let total_cycles = clear_count
        + receive_count
        + run_count
        + check_count
        + solution_count
        + split_count
        + pause_count;

    let percentage = |count: u64| 100.0 * count as f64 / total_cycles as f64;

    println!("Time spent in each state:");
    println!("  Clear:    {:5.2}%", percentage(clear_count));
    println!("  Receive:  {:5.2}%", percentage(receive_count));
    println!("  Run:      {:5.2}%", percentage(run_count));
    println!("  Check:    {:5.2}%", percentage(check_count));
    println!("  Solution: {:5.2}%", percentage(solution_count));
    println!("  Split:    {:5.2}%", percentage(split_count));
    println!("  Pause:    {:5.2}%", percentage(pause_count));

    Ok(())
}
