from amaranth import *
from amaranth.hdl import ValueLike


def pipe(m: Module, input: ValueLike) -> Signal:
    # src_loc_at tells Signal how far up in the call chain to look for what to name
    # the signal: so, setting it to 1 means we want it to use the name of the
    # variable the caller's assigning our result to.
    output = Signal.like(input, src_loc_at=1)
    m.d.sync += output.eq(input)
    return output
