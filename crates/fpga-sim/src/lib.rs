use std::cell::Cell;
use std::ffi::{c_char, c_void, CStr};
use std::fmt::Debug;
use std::iter::zip;

use net_finder::{fpga, FinderCtx, FinderInfo};

pub const MAX_AREA: usize = 64;
pub const MAX_CUBOIDS: usize = 3;

#[repr(C)]
struct InterfaceInner {
    in_payload: *const Cell<u8>,
    in_valid: *const Cell<bool>,
    in_ready: *const Cell<bool>,

    out_payload: *const Cell<u8>,
    out_valid: *const Cell<bool>,
    out_ready: *const Cell<bool>,

    req_pause: *const Cell<bool>,
    req_split: *const Cell<bool>,

    wants_finder: *const Cell<bool>,
    stepping: *const Cell<bool>,
}

#[repr(C)]
struct NeighbourLookupInner {
    en: *const Cell<bool>,
    addr: *const Cell<u16>,
    data: *const Cell<u64>,
}

#[repr(C)]
struct UndoLookupInner {
    en: *const Cell<bool>,
    addr: *const Cell<u16>,
    data: *const Cell<u16>,
}

#[repr(C)]
struct CoreInner {
    ptr: *mut c_void,

    // Invariant: outside of `Core::clock`, this is always `false`.
    clk: *const Cell<bool>,
    reset: *const Cell<bool>,

    interfaces: [InterfaceInner; 4],
    neighbour_lookups: [NeighbourLookupInner; 3],
    undo_lookups: [UndoLookupInner; 2],
}

extern "C" {
    // TODO: it doesn't look like Verilator uses C++ exceptions but it might not
    // hurt to make sure, because unwinding over FFI is UB.
    fn verilated_context_new(trace: bool) -> *mut c_void;
    fn verilated_context_time(this: *mut c_void) -> u64;
    fn verilated_context_time_inc(this: *mut c_void, add: u64);
    fn verilated_context_free(this: *mut c_void);

    fn verilated_vcd_new() -> *mut c_void;
    fn verilated_vcd_open(this: *mut c_void, filename: *const c_char);
    fn verilated_vcd_dump(this: *mut c_void, time: u64);
    fn verilated_vcd_flush(this: *mut c_void);
    fn verilated_vcd_free(this: *mut c_void);

    fn core_new(context: *mut c_void) -> CoreInner;
    fn core_update(this: *mut c_void);
    fn core_trace(this: *mut c_void, trace: *mut c_void);
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

    pub fn time(&self) -> u64 {
        unsafe { verilated_context_time(self.ptr) }
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
    vcd: Option<*mut c_void>,
    // This can be retrieved via `VerilatedModel::contextp`, but that gives a pointer to the
    // underlying `VerilatedContext` when `&VerilatedContext` in Rust is actually a pointer to a
    // pointer. So we'd probably have to add a `VerilatedContextRef`, and so on... this is easier.
    ctx: &'a VerilatedContext,
    /// The clock cycle number we're up to.
    tick: Cell<u64>,
    tick_event: event_listener::Event,
}

/// A simulated instance of `core`.
impl<'a> Core<'a> {
    /// Creates a new simulated instance of `core`.
    ///
    /// You can optionally provide the name of a file to write a VCD trace of
    /// the simulation to.
    // TODO: I'm pretty sure it's ok for this to be a shared reference but idk.
    pub fn new(ctx: &'a VerilatedContext, trace_file: Option<&CStr>) -> Self {
        let inner = unsafe { core_new(ctx.ptr) };
        let vcd = trace_file.map(|filename| unsafe {
            let ptr = verilated_vcd_new();
            core_trace(inner.ptr, ptr);
            verilated_vcd_open(ptr, filename.as_ptr());
            ptr
        });

        let this = Self {
            inner,
            vcd,
            ctx,
            tick: Cell::new(0),
            tick_event: event_listener::Event::new(),
        };

        // Initialise all the `core`'s inputs low.
        this.clk().set(false);
        this.reset_sig().set(false);

        for interface in this.interfaces() {
            interface.in_payload.set(0);
            interface.in_valid.set(false);
            interface.out_ready.set(false);
            interface.req_pause.set(false);
            interface.req_split.set(false);
        }

        for mem in this.neighbour_lookups() {
            mem.en.set(false);
            mem.addr.set(0);
            mem.data.set(0);
        }

        for mem in this.undo_lookups() {
            mem.en.set(false);
            mem.addr.set(0);
            mem.data.set(0);
        }

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
    pub fn update(&self) {
        self.ctx.time_inc(1);
        unsafe {
            core_update(self.inner.ptr);
            if let Some(ptr) = self.vcd {
                verilated_vcd_dump(ptr, self.ctx.time());
            }
        }
    }

    pub fn flush_vcd(&self) {
        if let Some(ptr) = self.vcd {
            unsafe { verilated_vcd_flush(ptr) }
        }
    }

    fn clk(&self) -> &Cell<bool> {
        unsafe { &*self.inner.clk }
    }
    fn reset_sig(&self) -> &Cell<bool> {
        unsafe { &*self.inner.reset }
    }

    /// Resets the `core`.
    pub fn reset(&self) {
        self.reset_sig().set(true);
        self.update();

        self.clock();

        self.reset_sig().set(false);
        self.update();
    }

    /// Runs a clock cycle of the `core`.
    pub fn clock(&self) {
        self.clk().set(true);
        self.update();
        self.clk().set(false);
        self.update();

        self.tick.set(self.tick.get() + 1);
        self.tick_event.notify(usize::MAX);
    }

    pub fn interfaces(&self) -> impl ExactSizeIterator<Item = Interface> + DoubleEndedIterator {
        self.inner.interfaces.iter().map(|raw| Interface {
            core: self,
            in_payload: unsafe { &*raw.in_payload },
            in_valid: unsafe { &*raw.in_valid },
            in_ready: unsafe { &*raw.in_ready },
            out_payload: unsafe { &*raw.out_payload },
            out_valid: unsafe { &*raw.out_valid },
            out_ready: unsafe { &*raw.out_ready },
            req_pause: unsafe { &*raw.req_pause },
            req_split: unsafe { &*raw.req_split },
            wants_finder: unsafe { &*raw.wants_finder },
            stepping: unsafe { &*raw.stepping },
        })
    }

    fn neighbour_lookups(
        &self,
    ) -> impl ExactSizeIterator<Item = NeighbourLookup> + DoubleEndedIterator {
        self.inner
            .neighbour_lookups
            .iter()
            .map(|raw| NeighbourLookup {
                core: self,
                en: unsafe { &*raw.en },
                addr: unsafe { &*raw.addr },
                data: unsafe { &*raw.data },
            })
    }

    fn undo_lookups(&self) -> impl ExactSizeIterator<Item = UndoLookup> + DoubleEndedIterator {
        self.inner.undo_lookups.iter().map(|raw| UndoLookup {
            core: self,
            en: unsafe { &*raw.en },
            addr: unsafe { &*raw.addr },
            data: unsafe { &*raw.data },
        })
    }

    /// Waits for a clock cycle to pass.
    ///
    /// Make sure not to `await` anything else: the fuzz target will perform a
    /// clock cycle whenever all its tasks are idle, so you might end up
    /// missing a clock cycle.
    pub async fn tick(&self) {
        let tick = self.tick.get();
        while self.tick.get() == tick {
            self.tick_event.listen().await;
        }
    }

    pub fn fill_mems<const CUBOIDS: usize>(&self, ctx: &FinderCtx<CUBOIDS>) {
        for (mem, contents) in zip(
            self.neighbour_lookups(),
            fpga::neighbour_lookups(ctx, MAX_AREA, MAX_CUBOIDS),
        ) {
            mem.fill(&contents);
        }

        for (mem, contents) in zip(self.undo_lookups(), fpga::undo_lookups(ctx, MAX_CUBOIDS)) {
            mem.fill(&contents);
        }
    }
}

impl Drop for Core<'_> {
    fn drop(&mut self) {
        unsafe {
            // Make sure to drop the `VerilatedVcdC` first.
            if let Some(ptr) = self.vcd {
                verilated_vcd_free(ptr);
            }
            core_free(self.inner.ptr);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FinderType {
    Solution = 0,
    Split = 1,
    Pause = 2,
}

pub struct Interface<'a> {
    core: &'a Core<'a>,

    in_payload: &'a Cell<u8>,
    in_valid: &'a Cell<bool>,
    in_ready: &'a Cell<bool>,

    out_payload: &'a Cell<u8>,
    pub out_valid: &'a Cell<bool>,
    out_ready: &'a Cell<bool>,

    pub req_pause: &'a Cell<bool>,
    pub req_split: &'a Cell<bool>,

    pub wants_finder: &'a Cell<bool>,
    stepping: &'a Cell<bool>,
}

impl Interface<'_> {
    pub fn core(&self) -> &Core {
        self.core
    }

    fn set_in_data(&self, value: bool) {
        self.in_payload
            .set(self.in_payload.get() & !1 | value as u8)
    }

    fn set_in_last(&self, value: bool) {
        self.in_payload
            .set(self.in_payload.get() & !(1 << 1) | ((value as u8) << 1))
    }

    fn out_data(&self) -> bool {
        self.out_payload.get() & 1 != 0
    }

    fn out_last(&self) -> bool {
        (self.out_payload.get() >> 1) & 1 != 0
    }

    fn out_type(&self) -> FinderType {
        match self.out_payload.get() >> 2 {
            0 => FinderType::Solution,
            1 => FinderType::Split,
            2 => FinderType::Pause,
            _ => unreachable!(),
        }
    }

    /// Clocks this `core` until a clock edge occurs where `self.stepping()` is
    /// true; that is, the core's list of decisions changed.
    ///
    /// Alternatively, it'll return if the core finishes running its finder, in
    /// which case this returns `false`.
    ///
    /// This will ignore any solutions the finder produces along the way.
    pub async fn step(&self) -> bool {
        loop {
            if self.in_ready.get() {
                return false;
            }
            // If `self.stepping()` was true before a clock edge, that means stepping
            // occured on that clock edge.
            let stepped = self.stepping.get();
            self.core.tick().await;
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
    pub async fn load_finder<const CUBOIDS: usize>(
        &self,
        ctx: &FinderCtx<CUBOIDS>,
        info: &FinderInfo<CUBOIDS>,
    ) {
        let finder_bits = info.to_bits(ctx, MAX_AREA, MAX_CUBOIDS);

        let expected_steps = info.decisions.iter().filter(|&&decision| decision).count();
        let mut steps = 0;

        for (i, &bit) in finder_bits.iter().enumerate() {
            // First set up the inputs and update the outputs.
            self.set_in_data(bit);
            self.in_valid.set(true);
            self.set_in_last(i == finder_bits.len() - 1);
            self.core.update();

            // Then clock the `core` until there's a clock edge where `in_ready` is 1, which
            // means that it consumed the bit.
            loop {
                let consumed = self.in_ready.get();
                if self.stepping.get() {
                    steps += 1;
                }
                self.core.tick().await;
                if consumed {
                    break;
                }
            }
        }

        // Reset everything we used back to 0.
        self.set_in_data(false);
        self.in_valid.set(false);
        self.set_in_last(false);
        self.core.update();

        // Wait until all the steps we told the core to do have actually been performed,
        // so they don't get misinterpreted as actual runtime steps.
        while steps < expected_steps {
            if self.stepping.get() {
                steps += 1;
            }
            self.core.tick().await;
        }
    }

    /// Receives a finder from the core.
    async fn recv_finder<const CUBOIDS: usize>(
        &self,
        ctx: &FinderCtx<CUBOIDS>,
    ) -> FinderInfo<CUBOIDS> {
        // First, receive the raw bits from the core.
        let mut finder_bits = Vec::new();
        self.out_ready.set(true);
        self.core.update();
        loop {
            if self.out_valid.get() {
                finder_bits.push(self.out_data());
            }
            let done = self.out_valid.get() && self.out_last();
            self.core.tick().await;
            if done {
                break;
            }
        }

        self.out_ready.set(false);
        self.core.update();

        // Then interpret them as a `FinderInfo`.
        FinderInfo::from_bits(ctx, MAX_AREA, MAX_CUBOIDS, finder_bits)
    }

    /// Waits for an 'event' to occur in the core, and returns it.
    pub async fn event<const CUBOIDS: usize>(&self, ctx: &FinderCtx<CUBOIDS>) -> Event<CUBOIDS> {
        loop {
            match (
                self.stepping.get(),
                self.out_valid.get(),
                self.wants_finder.get(),
            ) {
                (false, false, false) => {
                    // Nothing's happened, wait another clock cycle.
                    self.core.tick().await;
                }
                (true, false, false) => {
                    // A step has occured.
                    self.core.tick().await;
                    return Event::Step;
                }
                (false, true, false) => {
                    let finder = self.recv_finder(ctx).await;
                    return match self.out_type() {
                        // The core's outputting a solution, receive and return it.
                        FinderType::Solution => Event::Solution(finder),
                        // The core's splitting its finder in half, receive the second half and
                        // return it.
                        FinderType::Split => Event::Split(finder),
                        // The core's pausing, receive its finder and return it.
                        FinderType::Pause => Event::Pause(finder),
                    };
                }
                (false, false, true) => {
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
    pub async fn pause<const CUBOIDS: usize>(
        &self,
        ctx: &FinderCtx<CUBOIDS>,
    ) -> Option<FinderInfo<CUBOIDS>> {
        self.req_pause.set(true);
        self.core.update();

        loop {
            match self.event(ctx).await {
                Event::Step | Event::Solution(_) => {}
                Event::Split(_) => unreachable!("core splitted unprompted"),
                Event::Pause(info) => return Some(info),
                Event::Receiving => return None,
            }
        }
    }
}

pub struct NeighbourLookup<'a> {
    core: &'a Core<'a>,

    en: &'a Cell<bool>,
    addr: &'a Cell<u16>,
    data: &'a Cell<u64>,
}

impl NeighbourLookup<'_> {
    pub fn fill(&self, contents: &[u64]) {
        for (i, &entry) in contents.iter().enumerate() {
            self.en.set(true);
            self.addr.set(i.try_into().unwrap());
            self.data.set(entry);
            self.core.update();
            self.core.clock();
        }

        self.en.set(false);
        self.addr.set(0);
        self.data.set(0);
        self.core.update();
    }
}

pub struct UndoLookup<'a> {
    core: &'a Core<'a>,

    en: &'a Cell<bool>,
    addr: &'a Cell<u16>,
    data: &'a Cell<u16>,
}

impl UndoLookup<'_> {
    pub fn fill(&self, contents: &[u16]) {
        for (i, &entry) in contents.iter().enumerate() {
            self.en.set(true);
            self.addr.set(i.try_into().unwrap());
            self.data.set(entry);
            self.core.update();
            self.core.clock();
        }

        self.en.set(false);
        self.addr.set(0);
        self.data.set(0);
        self.core.update();
    }
}

/// An event returned by `Core::event`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Event<const CUBOIDS: usize> {
    /// The core's list of decisions has changed.
    Step,
    /// The core's found a possible solution; the state where it found that
    /// solution is given here.
    Solution(FinderInfo<CUBOIDS>),
    /// The core's split its finder in two; it's continuing to run one half, and
    /// the other half is given here.
    Split(FinderInfo<CUBOIDS>),
    /// The core's paused itself; it had the contained state when it did so.
    Pause(FinderInfo<CUBOIDS>),
    /// The core's finished running anything it was running and is now waiting
    /// to receive a new finder.
    Receiving,
}
