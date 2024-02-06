use std::fmt::Debug;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

use anyhow::anyhow;
use indicatif::ProgressBar;
use litex_bridge::{csr_struct, CsrGroup, CsrRo, CsrRw, SocInfo};
use net_finder::{Cuboid, Finder, FinderInfo, Runtime};
use wishbone_bridge::Bridge;

// TODO: add support for runtime-sized CSRs, because this is not always 6.
const FINDER_WORDS: usize = 6;

csr_struct! {
    struct CoreManagerRegisters<'a> {
        input: CsrRw<'a, FINDER_WORDS>,
        input_submit: CsrRw<'a>,
        output: CsrRo<'a, FINDER_WORDS>,
        output_consume: CsrRw<'a>,
        flow: CsrRo<'a>,
        active: CsrRo<'a>,
        req_pause: CsrRw<'a>,
    }
}

/// A `Runtime` which runs finders on an FPGA running a custom SoC.
pub struct FpgaRuntime<const CUBOIDS: usize> {
    /// Information about the SoC the FPGA's running.
    pub soc_info: SocInfo,
    /// The bridge we're using to communicate with the FPGA.
    pub bridge: Bridge,
    /// Whether only the SoC's CSRs are accessible via. the bridge, rather than
    /// its entire memory space.
    ///
    /// This is the case when communicating to it via. PCIe.
    pub csr_only: bool,
    /// The cuboids the SoC is hard-coded to find nets for (parsed from
    /// `soc_info` so that `Self::cuboids()` is infallible).
    pub cuboids: [Cuboid; CUBOIDS],
}

impl<const CUBOIDS: usize> Runtime<CUBOIDS> for FpgaRuntime<CUBOIDS> {
    fn cuboids(&self) -> [Cuboid; CUBOIDS] {
        self.cuboids
    }

    fn run(
        self,
        ctx: &net_finder::FinderCtx<CUBOIDS>,
        mut input_finders: &[FinderInfo<CUBOIDS>],
        solution_tx: &std::sync::mpsc::Sender<net_finder::Solution>,
        pause: &std::sync::atomic::AtomicBool,
        // TODO: add counters to the SoC so we can update the progress bar.
        _progress: Option<&ProgressBar>,
    ) -> anyhow::Result<Vec<FinderInfo<CUBOIDS>>> {
        let reset_addr = CsrRw::<1>::addrs(&self.soc_info, self.csr_only, "ctrl_reset")?;
        let reset = CsrRw::backed_by(&self.bridge, reset_addr);
        reset.write([1])?;
        thread::sleep(Duration::from_millis(10));

        let reg_addrs = CoreManagerRegisters::addrs(&self.soc_info, self.csr_only, "core_mgr")?;
        let regs = CoreManagerRegisters::backed_by(&self.bridge, reg_addrs);

        let square_bits = clog2(ctx.target_area);
        let cursor_bits = square_bits + 2;
        let mapping_bits = u32::try_from(self.cuboids.len())? * cursor_bits;
        let mapping_index_bits: u32 = ctx
            .outer_square_caches
            .iter()
            .map(|cache| clog2(cache.classes().len()))
            .sum();
        let decision_index_bits = clog2(4 * ctx.target_area);

        let prefix_bits = mapping_bits + mapping_index_bits + decision_index_bits;
        let finder_bits = prefix_bits + 4 * u32::try_from(ctx.target_area)?;
        let finder_len_bits = clog2(usize::try_from(finder_bits)? + 1);

        // The finders we've received back from the SoC since asking it to pause.
        let mut paused_finders = Vec::new();

        let mut flow = regs.flow().read()?[0];
        // Keep going as long as either we still have more finders to send to the SoC,
        // the SoC's still running, or the SoC still has some finders buffered up to
        // send to us.
        while !input_finders.is_empty() || regs.active().read()? == [1] || flow & 0b10 != 0 {
            if flow & 0b01 != 0 {
                // The SoC is ready to receive a finder, send one to it if there are still any
                // left.
                if let Some((finder, rest)) = input_finders.split_first() {
                    input_finders = rest;

                    let mut bits = finder.to_bits(ctx);
                    let len = bits.len();
                    bits.resize(finder_bits.try_into()?, false);
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
                            for (i, &bit) in chunk.iter().enumerate() {
                                word |= (bit as u32) << i;
                            }
                            word
                        })
                        .collect::<Vec<_>>()
                        .try_into()
                        .map_err(|_| anyhow!("input CSR was unexpected size"))?;
                    regs.input().write(input)?;
                    regs.input_submit().write([1])?;
                }
            }

            if flow & 0b10 != 0 {
                // The SoC has a finder ready for us to receive, read it.
                let mut data = regs.output().read()?;
                regs.output_consume().write([1])?;
                // Reverse the words of the CSR so that it's little-endian the whole way
                // through.
                data.reverse();

                let mut bits: Vec<bool> = (0..finder_bits + finder_len_bits + 1)
                    .map(|i| data[usize::try_from(i).unwrap() / 32] & (1 << (i % 32)) != 0)
                    .collect();

                let is_pause = bits.pop().unwrap();
                let len: usize = take_bits(
                    bits[usize::try_from(finder_bits)?..].iter().copied(),
                    finder_len_bits,
                );
                bits.truncate(finder_bits.try_into()?);

                // `from_bits` expects the bits to be in big-endian order, so we have to reverse
                // them.
                bits.reverse();
                bits.truncate(len);
                let info = FinderInfo::from_bits(ctx, bits);

                if is_pause {
                    paused_finders.push(info);
                } else {
                    // Create a finder from the received info and see if it yields any solutions.
                    let mut finder = Finder::new(ctx, &info)?;
                    for solution in finder.finish_and_finalize(ctx) {
                        solution_tx.send(solution)?;
                    }
                }
            }

            if pause.load(Ordering::Relaxed) {
                // We've been asked to pause, so ask the SoC to pause in turn.
                regs.req_pause().write([1])?;
            }

            flow = regs.flow().read()?[0];
        }

        // If we paused, this is the list of finders the SoC gave us back like it should
        // be; otherwise it shouldn't have ever sent us any paused finders, and so it'll
        // be empty like it should be.
        Ok(paused_finders)
    }
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
