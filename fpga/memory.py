from typing import Union

from amaranth import *
from amaranth import ShapeLike
from amaranth.lib import wiring
from amaranth.lib.memory import Memory, ReadPort, WritePort
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2, exact_log2


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
        # This internally asserts that depth is a power of 2.
        exact_log2(depth)

        self._sdp_ports: list[tuple[ReadPort, WritePort]] = []

        super().__init__({})

    def sdp_port(self) -> tuple[ReadPort, WritePort]:
        addr_width = ceil_log2(self._chunks * self._depth)

        # Return disconnected interfaces, which we then add to an array and hook up
        # during `elaborate`.
        read_port = ReadPort.Signature(
            addr_width=addr_width, shape=self._shape
        ).create()
        write_port = WritePort.Signature(
            addr_width=addr_width, shape=self._shape
        ).create()

        self._sdp_ports.append((read_port, write_port))

        return read_port, write_port

    def elaborate(self, platform) -> Module:
        m = Module()

        chunk_addr_width = exact_log2(self._depth)

        inner_sdp_read_ports = [[] for _ in self._sdp_ports]
        for chunk_index in range(self._chunks):
            chunk = Memory(shape=self._shape, depth=self._depth, init=[])
            # Register the chunk as a submodule.
            m.submodules[f"chunk{chunk_index}"] = chunk

            # Give the chunk a port corresponding to each of our outer ports, and hook up
            # their inputs.
            for port_index, (read_port, write_port) in enumerate(self._sdp_ports):
                inner_read_port = chunk.read_port()
                inner_write_port = chunk.write_port()

                m.d.comb += inner_write_port.en.eq(
                    write_port.en & (write_port.addr[chunk_addr_width:] == chunk_index)
                )
                m.d.comb += inner_write_port.data.eq(write_port.data)

                # Make sure that the read and write ports always use the same address so that
                # they can infer a single port of a true-dual-port BRAM.
                addr = Mux(inner_write_port.en, write_port.addr, read_port.addr)
                m.d.comb += inner_write_port.addr.eq(addr[:chunk_addr_width])
                m.d.comb += inner_read_port.addr.eq(addr[:chunk_addr_width])

                inner_sdp_read_ports[port_index].append(inner_read_port)

        # Connect up the read ports' outputs.
        for (read_port, _), inner_read_ports in zip(
            self._sdp_ports, inner_sdp_read_ports
        ):
            m.d.comb += read_port.data.eq(
                Array(inner_read_ports)[read_port.addr[chunk_addr_width:]].data,
            )

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

    def __init__(self, *, shape: ShapeLike, depth: int, ports_per_memory: int = 2):
        self._shape = shape
        self._depth = depth
        self._ports_per_memory = ports_per_memory

        self._read_ports: list[ReadPort] = []

        super().__init__(
            {
                "write_port": In(
                    WritePort.Signature(addr_width=ceil_log2(depth), shape=shape)
                )
            }
        )

    def read_port(self) -> ReadPort:
        read_port = ReadPort.Signature(
            addr_width=ceil_log2(self._depth), shape=self._shape
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
                memory = Memory(shape=self._shape, depth=self._depth, init=[])
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

        return m
