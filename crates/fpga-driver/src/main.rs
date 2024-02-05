use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::{fs, thread};

use anyhow::bail;
use chrono::{DateTime, Local};
use clap::Parser;
use litex_bridge::{csr_struct, CsrGroup, CsrRo, CsrRw, SocConstant, SocInfo};
use net_finder::{Cuboid, Finder, FinderCtx, FinderInfo, Net};
use wishbone_bridge::{EthernetBridge, PCIeBridge};

// TODO: add support for runtime-sized CSRs, because this is not always 6.
const FINDER_WORDS: usize = 6;

#[derive(Parser)]
#[command(group = clap::ArgGroup::new("bridge").required(true).multiple(false))]
struct Args {
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

    let Some(Some(SocConstant::String(cuboids))) =
        soc_info.constants.get(&Cow::Borrowed("config_cuboids"))
    else {
        bail!("unable to read cuboids")
    };

    let cuboids: Vec<Cuboid> = cuboids
        .split(";")
        .map(|s| s.parse::<Cuboid>())
        .collect::<Result<_, _>>()?;

    match cuboids.as_slice() {
        &[a] => main_inner(&soc_info, args.pcie_device, args.udp_ip, [a]),
        &[a, b] => main_inner(&soc_info, args.pcie_device, args.udp_ip, [a, b]),
        &[a, b, c] => main_inner(&soc_info, args.pcie_device, args.udp_ip, [a, b, c]),
        _ => bail!("only 1, 2, or 3 cuboids are currently supported"),
    }
}

csr_struct! {
    struct CoreManagerRegisters<'a> {
        input: CsrRw<'a, FINDER_WORDS>,
        input_submit: CsrRw<'a>,
        output: CsrRo<'a, FINDER_WORDS>,
        output_consume: CsrRw<'a>,
        flow: CsrRo<'a>,
        active: CsrRo<'a>,
    }
}

fn main_inner<const CUBOIDS: usize>(
    soc_info: &SocInfo,
    pcie_device: Option<PathBuf>,
    udp_ip: Option<String>,
    cuboids: [Cuboid; CUBOIDS],
) -> anyhow::Result<()> {
    let bridge = if let Some(ref device) = pcie_device {
        PCIeBridge::new(device.join("resource0"))?.create()?
    } else if let Some(ip) = udp_ip {
        EthernetBridge::new(ip)?.create()?
    } else {
        unreachable!()
    };
    bridge.connect()?;

    let reset_addr = CsrRw::<1>::addrs(soc_info, pcie_device.is_some(), "ctrl_reset")?;
    let reset = CsrRw::backed_by(&bridge, reset_addr);
    reset.write([1])?;
    thread::sleep(Duration::from_millis(10));

    let reg_addrs = CoreManagerRegisters::addrs(soc_info, pcie_device.is_some(), "core_mgr")?;
    let regs = CoreManagerRegisters::backed_by(&bridge, reg_addrs);

    let ctx = FinderCtx::new(cuboids, Duration::ZERO)?;

    let square_bits = clog2(ctx.target_area);
    let cursor_bits = square_bits + 2;
    let mapping_bits = u32::try_from(cuboids.len())? * cursor_bits;
    let mapping_index_bits: u32 = ctx
        .outer_square_caches
        .iter()
        .map(|cache| clog2(cache.classes().len()))
        .sum();
    let decision_index_bits = clog2(4 * ctx.target_area);

    let prefix_bits = mapping_bits + mapping_index_bits + decision_index_bits;
    let finder_bits = prefix_bits + 4 * u32::try_from(ctx.target_area)?;
    let finder_len_bits = clog2(usize::try_from(finder_bits).unwrap() + 1);

    let started = AtomicBool::new(false);
    thread::scope(|s| {
        s.spawn(|| {
            // TODO: resuming
            let finders = ctx.gen_finders();
            for finder in finders {
                let info = finder.into_info(&ctx);
                let mut bits = info.to_bits(&ctx);
                let len = bits.len();
                bits.resize(finder_bits.try_into().unwrap(), false);
                // `Finder::to_bits` outputs the bits in big-endian order, but from here on
                // we're dealing with little-endian so reverse them.
                bits.reverse();
                bits.extend(int_bits(len, finder_len_bits));

                let input: [u32; FINDER_WORDS] = bits
                    .chunks(32)
                    // The individual bits go in little-endian order, but the addresses go in
                    // big-endian order, so we have to reverse the order of the chunks.
                    .rev()
                    .map(|chunk| {
                        let mut word: u32 = 0;
                        for (j, &bit) in chunk.iter().enumerate() {
                            word |= (bit as u32) << j;
                        }
                        word
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                regs.input().write(input).unwrap();
                // Wait until the SoC's ready for another finder before submitting it.
                while regs.flow().read().unwrap()[0] & 0b01 == 0 {
                    std::hint::spin_loop();
                }
                regs.input_submit().write([1]).unwrap();
            }

            // Tell the reader that the SoC's started so that it can start checking `active`
            // (if it did so immediately we'd immediately exit because the SoC's inactive
            // until we send it finders).
            started.store(true, Ordering::Relaxed);
        });

        s.spawn(|| {
            let mut count = 0;
            let mut yielded_nets: HashSet<Net> = HashSet::new();
            let start = Instant::now();

            let mut wait_loops: u64 = 0;
            let mut total_wait_iters: u64 = 0;

            'outer: loop {
                // Wait until the SoC has a finder ready for us to read.
                wait_loops += 1;
                while regs.flow().read().unwrap()[0] & 0b10 == 0 {
                    // Note: it's important that we only check this if there's nothing ready for us
                    // to read, since it's perfectly possible for the FPGA to finish its work and
                    // `active` to become 0 before we've finished processing its outputs.
                    if started.load(Ordering::Relaxed) && regs.active().read().unwrap() == [0] {
                        break 'outer;
                    }
                    std::hint::spin_loop();
                    total_wait_iters += 1;
                }

                // Then read it.
                let mut data = regs.output().read().unwrap();
                regs.output_consume().write([1]).unwrap();
                // Reverse the words of the CSR so that it's little-endian the whole way
                // through.
                data.reverse();

                let mut bits: Vec<bool> = (0..finder_bits + finder_len_bits + 1)
                    .map(|i| data[usize::try_from(i).unwrap() / 32] & (1 << i % 32) != 0)
                    .collect();

                let is_pause = bits.pop().unwrap();
                let len: usize = take_bits(
                    bits[usize::try_from(finder_bits).unwrap()..]
                        .iter()
                        .copied(),
                    finder_len_bits,
                );
                bits.truncate(finder_bits.try_into().unwrap());

                // `from_bits` expects the bits to be in big-endian order, so we have to reverse
                // them.
                bits.reverse();
                bits.truncate(len);
                let info = FinderInfo::from_bits(&ctx, bits.iter().copied());

                // For now, we never pause so this should always be 0.
                assert!(!is_pause);

                // Create a finder from the received info and see if it yields any solutions.
                let mut finder = Finder::new(&ctx, &info).unwrap();
                for solution in finder.finish_and_finalize(&ctx) {
                    let new = yielded_nets.insert(solution.net.clone());
                    if !new {
                        continue;
                    }

                    count += 1;
                    println!(
                        "#{count} after {:.3?} ({}):",
                        solution.search_time,
                        DateTime::<Local>::from(solution.time).format("at %r on %e %b")
                    );
                    println!("{solution}");
                    assert!(cuboids
                        .into_iter()
                        .all(|cuboid| solution.net.color(cuboid).is_some()));
                }
            }

            println!("Number of nets: {count} (took {:?})", start.elapsed());
            println!(
                "Average number of loops spent waiting: {}",
                total_wait_iters as f64 / wait_loops as f64
            );
        });
    });

    Ok(())
}

fn clog2(x: usize) -> u32 {
    usize::BITS - (x - 1).leading_zeros()
}

// Note: these are different from the versions used to implement
// `FinderInfo::{from,to}_bits` because they're little-endian, not big-endian.

/// Returns an iterator over the lower `bits` bits of `int`, from LSB to
/// MSB.
fn int_bits(int: impl Into<usize>, bits: u32) -> impl Iterator<Item = bool> {
    let int: usize = int.into();
    (0..bits).map(move |bit| int & (1 << bit) != 0)
}

/// Consumes the first `bits` bits from `iter` and returns them as an
/// integer, with the first bit read being the LSB and the last bit
/// being the MSB.
fn take_bits<T>(iter: impl Iterator<Item = bool>, bits: u32) -> T
where
    T: TryFrom<usize>,
    T::Error: Debug,
{
    let mut result = 0;
    let mut i = 0;
    for bit in iter.take(bits.try_into().unwrap()) {
        result |= (bit as usize) << i;
        i += 1;
    }
    assert_eq!(i, bits);
    result.try_into().unwrap()
}
