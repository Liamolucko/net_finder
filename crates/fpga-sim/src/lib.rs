use std::array;
use std::ffi::c_void;
use std::fmt::Debug;

use net_finder::{Cuboid, Cursor, FinderCtx, FinderInfo, Mapping};

pub const CUBOIDS: [Cuboid; 3] = [
    Cuboid::new(1, 1, 11),
    Cuboid::new(1, 2, 7),
    Cuboid::new(1, 3, 5),
];
pub const NUM_CUBOIDS: usize = CUBOIDS.len();
pub const AREA: usize = CUBOIDS[0].surface_area();

#[repr(C)]
struct CoreInner {
    ptr: *mut c_void,

    // Invariant: outside of `Core::clock`, this is always `false`.
    clk: *mut bool,
    reset: *mut bool,

    in_data: *mut bool,
    in_valid: *mut bool,
    in_ready: *const bool,
    in_last: *mut bool,

    out_data: *const bool,
    out_valid: *const bool,
    out_ready: *mut bool,
    out_last: *const bool,

    out_solution: *const bool,
    out_split: *const bool,
    out_pause: *const bool,

    req_pause: *mut bool,
    req_split: *mut bool,

    stepping: *const bool,
}

extern "C" {
    fn verilated_context_new(trace: bool) -> *mut c_void;
    fn verilated_context_time_inc(this: *mut c_void, add: u64);
    fn verilated_context_free(this: *mut c_void);

    fn core_new(context: *mut c_void) -> CoreInner;
    fn core_update(this: *mut c_void);
    fn core_free(this: *mut c_void);
}

pub struct VerilatedContext {
    ptr: *mut c_void,
}

impl VerilatedContext {
    pub fn new(trace: bool) -> Self {
        Self {
            ptr: unsafe { verilated_context_new(trace) },
        }
    }

    pub fn time_inc(&self, add: u64) {
        unsafe { verilated_context_time_inc(self.ptr, add) }
    }
}

impl Drop for VerilatedContext {
    fn drop(&mut self) {
        unsafe { verilated_context_free(self.ptr) }
    }
}

pub struct Core<'a> {
    inner: CoreInner,
    // This can be retrieved via `VerilatedModel::contextp`, but that gives a pointer to the
    // underlying `VerilatedContext` when `&VerilatedContext` in Rust is actually a pointer to a
    // pointer. So we'd probably have to add a `VerilatedContextRef`, and so on... this is easier.
    ctx: &'a VerilatedContext,
}

/// A simulated instance of `core`.
impl<'a> Core<'a> {
    /// Creates a new simulated instance of `core`.
    // TODO: I'm pretty sure it's ok for this to be a shared reference but idk.
    pub fn new(ctx: &'a VerilatedContext) -> Self {
        let mut this = Self {
            inner: unsafe { core_new(ctx.ptr) },
            ctx,
        };

        // Initialise all the `core`'s inputs low.
        *this.clk() = false;
        *this.reset_sig() = false;
        *this.in_data() = false;
        *this.in_valid() = false;
        *this.in_last() = false;
        *this.out_ready() = false;
        *this.req_pause() = false;
        *this.req_split() = false;

        this.update();

        this
    }

    /// Simulates the `core`'s response to any change in inputs since the last
    /// time `update` was called.
    ///
    /// This also increments the simulation timestep by 1 so that each update
    /// can be seen in the waveform.
    ///
    /// Note that this doesn't create an implicit clock edge or anything like
    /// that; the clock is just another input.
    pub fn update(&mut self) {
        self.ctx.time_inc(1);
        unsafe { core_update(self.inner.ptr) }
    }

    fn clk(&mut self) -> &mut bool {
        unsafe { &mut *self.inner.clk }
    }
    fn reset_sig(&mut self) -> &mut bool {
        unsafe { &mut *self.inner.reset }
    }
    pub fn in_data(&mut self) -> &mut bool {
        unsafe { &mut *self.inner.in_data }
    }
    pub fn in_valid(&mut self) -> &mut bool {
        unsafe { &mut *self.inner.in_valid }
    }
    pub fn in_last(&mut self) -> &mut bool {
        unsafe { &mut *self.inner.in_last }
    }
    pub fn out_ready(&mut self) -> &mut bool {
        unsafe { &mut *self.inner.out_ready }
    }
    pub fn req_pause(&mut self) -> &mut bool {
        unsafe { &mut *self.inner.req_pause }
    }
    pub fn req_split(&mut self) -> &mut bool {
        unsafe { &mut *self.inner.req_split }
    }

    pub fn in_ready(&self) -> bool {
        unsafe { *self.inner.in_ready }
    }
    pub fn out_data(&self) -> bool {
        unsafe { *self.inner.out_data }
    }
    pub fn out_valid(&self) -> bool {
        unsafe { *self.inner.out_valid }
    }
    pub fn out_last(&self) -> bool {
        unsafe { *self.inner.out_last }
    }
    pub fn out_solution(&self) -> bool {
        unsafe { *self.inner.out_solution }
    }
    pub fn out_split(&self) -> bool {
        unsafe { *self.inner.out_split }
    }
    pub fn out_pause(&self) -> bool {
        unsafe { *self.inner.out_pause }
    }
    pub fn stepping(&self) -> bool {
        unsafe { *self.inner.stepping }
    }

    /// Resets the `core`.
    pub fn reset(&mut self) {
        *self.reset_sig() = true;
        self.update();
        *self.reset_sig() = false;
        self.update();
    }

    /// Runs a clock cycle of the `core`.
    pub fn clock(&mut self) {
        *self.clk() = true;
        self.update();
        *self.clk() = false;
        self.update();
    }

    /// Clocks this `core` until a clock edge occurs where `self.stepping()` is
    /// true; that is, the core's list of decisions changed.
    ///
    /// Alternatively, it'll return if the core finishes running its finder, in
    /// which case this returns `false`.
    ///
    /// This will ignore any solutions the finder produces along the way.
    pub fn step(&mut self) -> bool {
        loop {
            if self.in_ready() {
                return false;
            }
            // If `self.stepping()` was true before a clock edge, that means stepping
            // occured on that clock edge.
            let stepped = self.stepping();
            self.clock();
            if stepped {
                break;
            }
        }
        true
    }

    /// Loads a `FinderInfo` into the `core`.
    ///
    /// If the `core` is already running a finder, this will get stuck in an
    /// infinite loop.
    pub fn load_finder(&mut self, ctx: &FinderCtx<NUM_CUBOIDS>, info: &FinderInfo<NUM_CUBOIDS>) {
        let start_mapping = info.start_mapping.sample(&ctx.outer_square_caches);

        /// Returns an iterator over the lower `bits` bits of `int`, from MSB to
        /// LSB.
        fn bits(int: impl Into<usize>, bits: u32) -> impl Iterator<Item = bool> {
            let int: usize = int.into();
            (0..bits).rev().map(move |bit| int & (1 << bit) != 0)
        }

        // First figure out the bits we need to actually send to the core.
        let mut finder_bits = Vec::new();
        // `mapping_t` on the FPGA is a packed array, which means higher indices come
        // first and so we need to add these in reverse order.
        for cursor in start_mapping.cursors.into_iter().rev() {
            finder_bits.extend(bits(cursor.0, clog2(4 * AREA)));
        }
        // Mapping indexes on the FPGA don't work the same way as the ones on
        // the CPU: they don't include the class of the cursor on the fixed
        // cuboid.
        for (cuboid, class) in info
            .start_mapping
            .classes
            .into_iter()
            .enumerate()
            .filter(|&(cuboid, _)| cuboid != ctx.fixed_cuboid)
        {
            finder_bits.extend(bits(
                class.index(),
                clog2(ctx.outer_square_caches[cuboid].classes().len()),
            ));
        }
        finder_bits.extend(bits(info.base_decision, clog2(4 * AREA)));

        finder_bits.extend(&info.decisions);

        // Then do that.
        for (i, &bit) in finder_bits.iter().enumerate() {
            // First set up the inputs and update the outputs.
            *self.in_data() = bit;
            *self.in_valid() = true;
            *self.in_last() = i == finder_bits.len() - 1;
            self.update();

            // Then clock the `core` until there's a clock edge where `in_ready` is 1, which
            // means that it consumed the bit.
            loop {
                let consumed = self.in_ready();
                self.clock();
                if consumed {
                    break;
                }
            }
        }

        // Reset everything we used back to 0.
        *self.in_data() = false;
        *self.in_valid() = false;
        *self.in_last() = false;
        self.update();
    }

    /// Receives a finder from the core.
    fn recv_finder(&mut self, ctx: &FinderCtx<NUM_CUBOIDS>) -> FinderInfo<NUM_CUBOIDS> {
        // First, receive the raw bits from the core.
        let mut finder_bits = Vec::new();
        *self.out_ready() = true;
        self.update();
        loop {
            if self.out_valid() {
                finder_bits.push(self.out_data());
            }
            let done = self.out_valid() && self.out_last();
            self.clock();
            if done {
                break;
            }
        }

        // Then interpret them as a `FinderInfo`.
        let mut finder_bits = finder_bits.into_iter();

        /// Consumes the first `bits` bits from `iter` and returns them as an
        /// integer, with the first bit read being the MSB and the last bit
        /// being the LSB.
        fn take_bits<T>(iter: impl Iterator<Item = bool>, bits: u32) -> T
        where
            T: TryFrom<usize>,
            T::Error: Debug,
        {
            let mut result = 0;
            let mut taken = 0;
            for bit in iter.take(bits.try_into().unwrap()) {
                result <<= 1;
                result |= bit as usize;
                taken += 1;
            }
            assert_eq!(taken, bits);
            result.try_into().unwrap()
        }

        let mut start_mapping = Mapping {
            cursors: array::from_fn(|_| Cursor(take_bits(&mut finder_bits, clog2(4 * AREA)))),
        };
        // `mapping_t` goes from highest index to lowest, so we need to reverse it.
        start_mapping.cursors.reverse();

        // Skip over `start_mapping_index`, we don't need it.
        take_bits::<usize>(
            &mut finder_bits,
            ctx.outer_square_caches
                .iter()
                .enumerate()
                .filter(|&(cuboid, _)| cuboid != ctx.fixed_cuboid)
                .map(|(_, cache)| clog2(cache.classes().len()))
                .sum(),
        );

        FinderInfo {
            start_mapping: start_mapping.to_classes(&ctx.outer_square_caches),
            base_decision: take_bits(&mut finder_bits, clog2(4 * AREA)),
            decisions: finder_bits.collect(),
        }
    }

    /// Waits for an 'event' to occur in the core, and returns it.
    pub fn event(&mut self, ctx: &FinderCtx<NUM_CUBOIDS>) -> Event {
        loop {
            match (
                self.stepping(),
                self.out_solution(),
                self.out_split(),
                self.out_pause(),
                self.in_ready(),
            ) {
                (false, false, false, false, false) => {
                    // Nothing's happened, wait another clock cycle.
                    self.clock();
                }
                (true, false, false, false, false) => {
                    // A step has occured.
                    self.clock();
                    return Event::Step;
                }
                (false, true, false, false, false) => {
                    // The core's outputting a solution, receive and return it.
                    return Event::Solution(self.recv_finder(ctx));
                }
                (false, false, true, false, false) => {
                    // The core's splitting its finder in half, receive the second half and return
                    // it.
                    return Event::Split(self.recv_finder(ctx));
                }
                (false, false, false, true, false) => {
                    // The core's pausing, receive its finder and return it.
                    return Event::Pause(self.recv_finder(ctx));
                }
                (false, false, false, false, true) => {
                    // The core's stopped (or had already stopped).
                    return Event::Receiving;
                }
                _ => unreachable!("more than one event happening simultaneously"),
            }
        }
    }

    /// Pauses this `core` and returns the finder it was running, or `None` if
    /// it wasn't running anything.
    ///
    /// Any solutions the `core` produces before pausing are ignored.
    pub fn pause(&mut self, ctx: &FinderCtx<NUM_CUBOIDS>) -> Option<FinderInfo<NUM_CUBOIDS>> {
        *self.req_pause() = true;
        self.update();

        loop {
            match self.event(ctx) {
                Event::Step | Event::Solution(_) => {}
                Event::Split(_) => unreachable!("core splitted unprompted"),
                Event::Pause(info) => return Some(info),
                Event::Receiving => return None,
            }
        }
    }
}

impl Drop for Core<'_> {
    fn drop(&mut self) {
        unsafe { core_free(self.inner.ptr) }
    }
}

/// An event returned by `Core::event`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Event {
    /// The core's list of decisions has changed.
    Step,
    /// The core's found a possible solution; the state where it found that
    /// solution is given here.
    Solution(FinderInfo<NUM_CUBOIDS>),
    /// The core's split its finder in two; it's continuing to run one half, and
    /// the other half is given here.
    Split(FinderInfo<NUM_CUBOIDS>),
    /// The core's paused itself; it had the contained state when it did so.
    Pause(FinderInfo<NUM_CUBOIDS>),
    /// The core's finished running anything it was running and is now waiting
    /// to receive a new finder.
    Receiving,
}

fn clog2(x: usize) -> u32 {
    usize::BITS - (x - 1).leading_zeros()
}
