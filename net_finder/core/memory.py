from itertools import chain

from amaranth import *
from amaranth.hdl import ShapeLike, ValueLike
from amaranth.lib import wiring
from amaranth.lib.memory import Memory, MemoryData, ReadPort, WritePort
from amaranth.lib.wiring import In, Out, PureInterface
from amaranth.utils import ceil_log2


class ChunkedReadPortSignature(wiring.Signature):
    def __init__(self, *, chunk_width: int, addr_width: int, shape: ShapeLike):
        super().__init__(
            {
                # I'm not bothering with `en`, I don't need it.
                "chunk": In(chunk_width),
                "addr": In(addr_width),
                "data": Out(shape),
            }
        )


class ChunkedWritePortSignature(wiring.Signature):
    def __init__(self, *, chunk_width: int, addr_width: int, shape: ShapeLike):
        super().__init__(
            {
                "en": In(1),
                "chunk": In(chunk_width),
                "addr": In(addr_width),
                "data": In(shape),
            }
        )


class ChunkedMemory(wiring.Component):
    """
    A wrapper around a collection of memories ('chunks') that behaves like a single
    memory.

    The reason for this type's existence is the `sdp_port` method, which provides a
    simple-dual-port interface implemented using a single port that works as long as
    its read and write ports are never used on the same chunk at the same time.

    This assumption is correct in our use case (accessing the net) because each
    finder gets its own chunk, is only being processed by one pipeline stage at a
    time, and the read/write ports are used in separate pipeline stages: thus the
    read and write ports will always be accessing different finders' chunks.
    """

    def __init__(self, *, shape: ShapeLike, depth: int, chunks: int):
        """
        Creates a new `ChunkedMemory`.

        Note that `depth` is the depth of each chunk, not of the entire memory.
        """

        self._shape = shape
        self._depth = depth
        self._chunks = chunks

        self._read_ports: list[tuple[PureInterface, str]] = []
        self._write_ports: list[PureInterface] = []
        self._sdp_ports: list[tuple[tuple[PureInterface, str], PureInterface]] = []

        super().__init__({})

    def read_port(self, domain="sync") -> PureInterface:
        # Return a disconnected interface, which we then add to an array and hook up
        # during `elaborate`.
        port = ChunkedReadPortSignature(
            chunk_width=ceil_log2(self._chunks),
            addr_width=ceil_log2(self._depth),
            shape=self._shape,
        ).create()

        self._read_ports.append((port, domain))

        return port

    def write_port(self) -> PureInterface:
        # Return a disconnected interface, which we then add to an array and hook up
        # during `elaborate`.
        port = ChunkedWritePortSignature(
            chunk_width=ceil_log2(self._chunks),
            addr_width=ceil_log2(self._depth),
            shape=self._shape,
        ).create()

        self._write_ports.append(port)

        return port

    def sdp_port(self, read_domain="sync") -> tuple[PureInterface, PureInterface]:
        # Return disconnected interfaces, which we then add to an array and hook up
        # during `elaborate`.
        read_port = ChunkedReadPortSignature(
            chunk_width=ceil_log2(self._chunks),
            addr_width=ceil_log2(self._depth),
            shape=self._shape,
        ).create()
        write_port = ChunkedWritePortSignature(
            chunk_width=ceil_log2(self._chunks),
            addr_width=ceil_log2(self._depth),
            shape=self._shape,
        ).create()

        self._sdp_ports.append(((read_port, read_domain), write_port))

        return read_port, write_port

    def elaborate(self, platform) -> Module:
        m = Module()

        # For each SDP port, the list of inner read ports it uses to access different
        # chunks.
        inner_sdp_read_ports = [[] for _ in self._sdp_ports]
        inner_read_ports = [[] for _ in self._read_ports]
        for chunk_index in range(self._chunks):
            chunk = Memory(
                shape=self._shape,
                depth=self._depth,
                init=[],
                attrs={"ram_style": "distributed"},
            )
            # Register the chunk as a submodule.
            m.submodules[f"chunk{chunk_index}"] = chunk

            # Give the chunk a port corresponding to each of our outer ports, and hook up
            # their inputs.
            for port_index, ((read_port, read_domain), write_port) in enumerate(
                self._sdp_ports
            ):
                inner_read_port = chunk.read_port(domain=read_domain)
                inner_write_port = chunk.write_port()

                m.d.comb += inner_write_port.en.eq(
                    write_port.en & (write_port.chunk == chunk_index)
                )
                m.d.comb += inner_write_port.data.eq(write_port.data)

                # Make sure that the read and write ports always use the same address so that
                # they can infer a single port of a true-dual-port BRAM.
                addr = Mux(inner_write_port.en, write_port.addr, read_port.addr)
                m.d.comb += inner_write_port.addr.eq(addr)
                m.d.comb += inner_read_port.addr.eq(addr)

                inner_sdp_read_ports[port_index].append(inner_read_port)

            for port_index, (port, domain) in enumerate(self._read_ports):
                inner_port = chunk.read_port(domain=domain)
                m.d.comb += inner_port.addr.eq(port.addr)
                inner_read_ports[port_index].append(inner_port)

            for port_index, port in enumerate(self._write_ports):
                inner_port = chunk.write_port()
                m.d.comb += inner_port.addr.eq(port.addr)
                m.d.comb += inner_port.data.eq(port.data)
                m.d.comb += inner_port.en.eq(port.en & (port.chunk == chunk_index))

        # Connect up the read ports' outputs.
        for (port, domain), inner_ports in zip(
            chain(self._read_ports, (r for r, _ in self._sdp_ports)),
            chain(inner_read_ports, inner_sdp_read_ports),
        ):
            chunk = Signal.like(port.chunk)
            m.d[domain] += chunk.eq(port.chunk)
            m.d.comb += port.data.eq(Array(inner_ports)[chunk].data)

        return m


class ConfigMemory(wiring.Component):
    """
    A wrapper around a memory (well, several replicas of the same memory) that
    allows having 1 write port and many read ports, with the same cost as if there
    were no write ports, as long as the read ports are never used at the same time
    as the write port.

    It accomplishes this by using the first port on each replica of the memory as
    both a read port and a write port, and setting its address to whichever half is
    being used.

    By default it gives each replica 2 ports (since that matches Xilinx 7 series
    BRAMs); you can alter this by setting `ports_per_memory` to a different value.
    I'm not aware of any scenarios where that's an improvement though (I expected 4
    to work better for 7 series distributed RAM, but it ends up inferring dual-port
    RAMs anyway).
    """

    def __init__(
        self,
        data: MemoryData | None = None,
        *,
        shape: ShapeLike | None = None,
        depth: int | None = None,
        init: list[ValueLike] | None = None,
        ports_per_memory: int = 2,
    ):
        # Copied from `Memory.__init__`.
        if data is None:
            if shape is None:
                raise ValueError("Either 'data' or 'shape' needs to be given")
            if depth is None:
                raise ValueError("Either 'data' or 'depth' needs to be given")
            if init is None:
                raise ValueError("Either 'data' or 'init' needs to be given")
            data = MemoryData(shape=shape, depth=depth, init=init)
        else:
            if not isinstance(data, MemoryData):
                raise TypeError(f"'data' must be a MemoryData instance, not {data!r}")
            if shape is not None:
                raise ValueError("'data' and 'shape' cannot be given at the same time")
            if depth is not None:
                raise ValueError("'data' and 'depth' cannot be given at the same time")
            if init is not None:
                raise ValueError("'data' and 'init' cannot be given at the same time")

        self._data = data
        self._ports_per_memory = ports_per_memory

        self._read_ports: list[ReadPort] = []

        super().__init__(
            {
                "write_port": Out(
                    WritePort.Signature(
                        addr_width=ceil_log2(data.depth), shape=data.shape
                    )
                )
            }
        )

    def read_port(self) -> ReadPort:
        read_port = ReadPort.Signature(
            addr_width=ceil_log2(self._data.depth), shape=self._data.shape
        ).create()
        self._read_ports.append(read_port)
        return read_port

    def elaborate(self, platform) -> Module:
        m = Module()

        memory = None
        memory_ports = 0
        memory_index = 0

        for read_port in self._read_ports:
            if memory is None:
                memory = Memory(self._data)
                m.submodules[f"memory{memory_index}"] = memory
                inner_write_port = memory.write_port()
            else:
                inner_write_port = None

            inner_read_port = memory.read_port()
            memory_ports += 1

            if memory_ports == self._ports_per_memory:
                memory = None
                memory_ports = 0
                memory_index += 1

            if inner_write_port is not None:
                # Make sure that the read and write ports always use the same address so that
                # they can infer a single port.
                addr = Mux(self.write_port.en, self.write_port.addr, read_port.addr)
                m.d.comb += inner_read_port.addr.eq(addr)
                m.d.comb += inner_write_port.addr.eq(addr)

                m.d.comb += inner_write_port.data.eq(self.write_port.data)
                m.d.comb += inner_write_port.en.eq(self.write_port.en)
            else:
                m.d.comb += inner_read_port.addr.eq(read_port.addr)

            m.d.comb += read_port.data.eq(inner_read_port.data)
            # TODO: handle read_port.en

        return m
