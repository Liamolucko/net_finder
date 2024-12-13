use std::fmt::Debug;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

use indicatif::ProgressBar;
use litex_bridge::{csr_struct, CsrGroup, CsrRo, CsrRw, DynCsrRo, DynCsrRw, SocInfo};
use net_finder::fpga::{self, clog2, max_decisions_len};
use net_finder::{Cuboid, Finder, FinderInfo, Runtime};
use wishbone_bridge::Bridge;

csr_struct! {
    pub struct CoreManagerRegisters<'a> {
        input: DynCsrRw<'a>,
        input_submit: CsrRw<'a>,
        output: DynCsrRo<'a>,
        output_consume: CsrRw<'a>,
        flow: CsrRo<'a>,
        neighbour_lookups_addr: CsrRw<'a>,
        neighbour_lookups_data: CsrRw<'a, 2>,
        neighbour_lookups_sel: CsrRw<'a>,
        undo_lookups_addr: CsrRw<'a>,
        undo_lookups_data: CsrRw<'a>,
        undo_lookups_sel: CsrRw<'a>,
        active: CsrRo<'a>,
        req_pause: CsrRw<'a>,
        split_finders: CsrRo<'a>,
        completed_finders: CsrRo<'a>,
        clear_count: CsrRo<'a, 2>,
        receive_count: CsrRo<'a, 2>,
        run_count: CsrRo<'a, 2>,
        check_count: CsrRo<'a, 2>,
        solution_count: CsrRo<'a, 2>,
        split_count: CsrRo<'a, 2>,
        pause_count: CsrRo<'a, 2>,
    }
}

const MAX_CUBOIDS: usize = 3;
const MAX_AREA: usize = 64;

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
    /// The cuboids we're solving for.
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
        progress: Option<&ProgressBar>,
    ) -> anyhow::Result<Vec<FinderInfo<CUBOIDS>>> {
        let reset_addr = CsrRw::<1>::addrs(&self.soc_info, self.csr_only, "ctrl_reset")?;
        let reset = CsrRw::backed_by(&self.bridge, reset_addr);
        reset.write([1])?;
        thread::sleep(Duration::from_millis(10));

        let reg_addrs = CoreManagerRegisters::addrs(&self.soc_info, self.csr_only, "core_mgr")?;
        let regs = CoreManagerRegisters::backed_by(&self.bridge, reg_addrs);

        for (i, contents) in fpga::neighbour_lookups(ctx, MAX_AREA, MAX_CUBOIDS)
            .into_iter()
            .enumerate()
        {
            for (j, entry) in contents.into_iter().enumerate() {
                regs.neighbour_lookups_addr()
                    .write([j.try_into().unwrap()])?;
                regs.neighbour_lookups_data()
                    .write([(entry >> 32) as u32, entry as u32])?;
                regs.neighbour_lookups_sel()
                    .write([i.try_into().unwrap()])?;
            }
        }

        for (i, contents) in fpga::undo_lookups(ctx, MAX_CUBOIDS).into_iter().enumerate() {
            for (j, entry) in contents.into_iter().enumerate() {
                regs.undo_lookups_addr().write([j.try_into().unwrap()])?;
                regs.undo_lookups_data().write([entry.into()])?;
                regs.undo_lookups_sel().write([i.try_into().unwrap()])?;
            }
        }

        let area_bits = clog2(MAX_AREA + 1);
        let square_bits = clog2(MAX_AREA);
        let cursor_bits = square_bits + 2;
        let mapping_bits = u32::try_from(MAX_CUBOIDS)? * cursor_bits;
        let class_bits = clog2(MAX_AREA);
        let mapping_index_bits = u32::try_from(MAX_CUBOIDS - 1)? * class_bits;
        let decision_index_bits = clog2(max_decisions_len(MAX_AREA));

        let prefix_bits = area_bits + mapping_bits + mapping_index_bits + decision_index_bits;
        let finder_bits = prefix_bits + u32::try_from(max_decisions_len(MAX_AREA))?;
        let finder_len_bits = clog2(usize::try_from(finder_bits)? + 1);

        let num_input_finders = u64::try_from(input_finders.len())?;
        // The finders we've received back from the SoC since asking it to pause.
        let mut paused_finders = Vec::new();

        let mut flow = regs.flow().read()?[0];
        // Keep going as long as either we still have more finders to send to the SoC,
        // the SoC's still running, or the SoC still has some finders buffered up to
        // send to us.
        while !input_finders.is_empty() || regs.active().read()? == [1] || flow & 0b10 != 0 {
            let mut did_something = false;

            if flow & 0b01 != 0 {
                // The SoC is ready to receive a finder, send one to it if there are still any
                // left.
                if let Some((finder, rest)) = input_finders.split_first() {
                    input_finders = rest;

                    let mut bits = finder.to_bits(ctx, MAX_AREA, MAX_CUBOIDS);
                    let len = bits.len();
                    bits.resize(finder_bits.try_into()?, false);
                    // `Finder::to_bits` outputs the bits in big-endian order, but from here on
                    // we're dealing with little-endian so reverse them.
                    bits.reverse();
                    bits.extend(int_bits(len, finder_len_bits));

                    let input: Vec<u32> = bits
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
                        .collect();
                    regs.input().write(&input)?;
                    regs.input_submit().write([1])?;

                    did_something = true;
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
                let info = FinderInfo::from_bits(ctx, MAX_AREA, MAX_CUBOIDS, bits);

                if is_pause {
                    paused_finders.push(info);
                } else {
                    // Create a finder from the received info and see if it yields any solutions.
                    let mut finder = Finder::new(ctx, &info)?;
                    for solution in finder.finish_and_finalize(ctx) {
                        solution_tx.send(solution)?;
                    }
                }

                did_something = true;
            }

            if pause.load(Ordering::Relaxed) {
                // We've been asked to pause, so ask the SoC to pause in turn.
                regs.req_pause().write([1])?;
            }

            if let Some(progress) = progress {
                progress.set_position(regs.completed_finders().read()?[0].into());
                progress.set_length(num_input_finders + u64::from(regs.split_finders().read()?[0]));
            }

            // Don't max out a CPU core doing nothing.
            if !did_something {
                thread::sleep(Duration::from_millis(5));
            }

            flow = regs.flow().read()?[0];
        }

        // If we paused, this is the list of finders the SoC gave us back like it should
        // be; otherwise it shouldn't have ever sent us any paused finders, and so it'll
        // be empty like it should be.
        Ok(paused_finders)
    }
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
