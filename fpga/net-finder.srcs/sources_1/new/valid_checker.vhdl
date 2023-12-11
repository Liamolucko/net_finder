library ieee;
use ieee.std_logic_1164.all;

-- A single-port, 1-bit wide RAM with asynchronous read and synchronous write.
--
-- You can optionally add extra read ports.
entity async_ram is
    generic(size: integer);
    port(
        clk: in std_logic;
        addr: in integer range 0 to size - 1;

        wr_data: in std_logic;
        wr_en: in std_logic;

        rd_data: out std_logic
    );
end async_ram;

architecture arch of async_ram is
    -- We can't just use a `std_logic_vector` here because Vivado doesn't infer
    -- memory if we do that: it only infers memory if you use an actual `array`.
    type buf is array(0 to size - 1) of std_logic;
    signal memory: buf;
begin
    rd_data <= memory(addr);
    process(clk)
    begin
        if clk'event and clk = '1' and wr_en = '1' then
            memory(addr) <= wr_data;
        end if;
    end process;
end arch;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.generated.all;

-- A RAM containing 1 bit for every square on a cuboid's surface.
--
-- It has 1 port supporting both asynchronous reads and synchronous writes, and
-- 4 ports that only support asynchronous reads.
entity surface is
    port(
        clk: in std_logic;

        -- The read/write port.
        rw_addr: in square;
        rw_wr_data: in std_logic;
        rw_wr_en: in std_logic;
        rw_rd_data: out std_logic;

        -- The read-only ports.
        rd_addrs: in square_vector(0 to 3);
        rd_data: out std_logic_vector(0 to 3)
    );
end surface;

architecture arch of surface is
    type buf is array(0 to area - 1) of std_logic;
    signal memory: buf;
begin
    rw_rd_data <= memory(rw_addr);
    gen: for i in 0 to 3 generate
        rd_data(i) <= memory(rd_addrs(i));
    end generate;

    process(clk)
    begin
        if clk'event and clk = '1' and rw_wr_en = '1' then
            memory(rw_addr) <= rw_wr_data;
        end if;
    end process;
end arch;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.generated.all;

-- An entity which checks which of an instruction's neighbours are valid, and
-- also maintains all the state necessary for doing so.
entity valid_checker is
    port(
        clk: in std_logic;

        -- The instruction to check the neighbours of.
        instruction: in instruction;
        start_mapping_index: in mapping_index;

        -- Whether `instruction` is being run and this `valid_checker`'s state needs to
        -- be updated accordingly.
        run: in std_logic;
        -- Whether `instruction` is being backtracked and this `valid_checker`'s state
        -- needs to be updated accordingly.
        backtrack: in std_logic;

        -- Whether `instruction`'s neighbour in each direction is valid.
        neighbours_valid: buffer std_logic_vector(0 to 3);
        -- Whether `instruction` itself is valid.
        --
        -- This only checks whether it tries to set any squares that are already
        -- filled. The instruction was valid back when we were processing its
        -- parent, which means it already has first dibs on its net position and
        -- whether it's skipped never changes.
        instruction_valid: out std_logic;

        -- 7 series distributed RAM doesn't have any way of clearing the whole thing at
        -- once, so we have to go through and individually set each bit to 0.
        clear_mode: in std_logic;
        clear_index: in integer range 0 to net_len / 4 - 1
    );
end valid_checker;

architecture arch of valid_checker is
    component async_ram is
        generic(size: integer);
        port(
            clk: in std_logic;
            addr: in integer range 0 to size - 1;
    
            wr_data: in std_logic;
            wr_en: in std_logic;
    
            rd_data: out std_logic
        );
    end component;

    component surface is
        port(
            clk: in std_logic;
    
            -- The read/write port.
            rw_addr: in square;
            rw_wr_data: in std_logic;
            rw_wr_en: in std_logic;
            rw_rd_data: out std_logic;

            -- The read-only ports.
            rd_addrs: in square_vector(0 to 3);
            rd_data: out std_logic_vector(0 to 3)
        );
    end component;

    -- The neighbours of `instruction` in each direction.
    signal neighbours: instruction_vector(0 to 3);

    -- For each neighbour of `instruction`, whether there's already a queued
    -- instruction setting the same net position.
    signal queued: std_logic_vector(0 to 3);
    -- Whether each neighbour of `instruction` tries to set any squares on the
    -- surface that are already filled.
    signal filled: std_logic_vector(0 to 3);
    -- Whether each neighbour of `instruction` is skipped due to being covered by
    -- another finder.
    signal skipped: std_logic_vector(0 to 3);

    type shard_num_vector is array(integer range <>) of integer range 0 to 3;
    type shard_index_vector is array(integer range <>) of integer range 0 to net_len / 4 - 1;
    -- Which shard of the net each neighbour's net position falls into.
    signal shards: shard_num_vector(0 to 3);
    -- What index we're looking at in each net shard.
    signal shard_indices: shard_index_vector(0 to 3);
    -- Whether we want to write to each net.
    --
    -- Unlike the surfaces, this varies between them because only the neighbours
    -- which were valid should be marked (or unmarked) as queued.
    signal shard_wr_ens: std_logic_vector(0 to 3);
    -- What value we've retrieved from each net shard.
    signal shard_values: std_logic_vector(0 to 3);

    type surface_values_vector is array(integer range <>) of std_logic_vector(0 to 3);
    type surface_indices_vector is array(integer range <>) of square_vector(0 to 3);

    -- What index the read/write port into each surface is using.
    signal surface_rw_indices: square_vector(0 to cuboids - 1);
    -- The value read by the read/write port into each surface.
    signal surface_rw_values: std_logic_vector(0 to cuboids - 1);
    -- For each surface, the indices that its read ports are using.
    signal surface_rd_indices: surface_indices_vector(0 to cuboids - 1);
    -- The values read by all the read ports of each surface.
    signal surface_rd_values: surface_values_vector(0 to cuboids - 1);

    -- Whether we want to write to the surfaces.
    signal surface_wr_en: std_logic;
    -- The value we want to write to the net & surfaces.
    --
    -- This is always the same between them because it's just 1 if we're running an
    -- instruction, and 0 otherwise - things get filled in when they get run, on
    -- both the net and surfaces.
    signal wr_data: std_logic;
begin
    gen_dirs: for direction in 0 to 3 generate
        neighbour_lookups: neighbour_lookup port map(
            instruction => instruction,
            direction => to_unsigned(direction, 2),
            neighbour => neighbours(direction)
        );

        with unsigned'(neighbours(direction).pos.x(1 downto 0) & neighbours(direction).pos.y(1 downto 0)) select
            shards(direction) <=
                -- Each 4x4 chunk of the net is split up into shards like this:
                --     2301
                --     2301
                --     0123
                --     0123
                -- where each number is the shard that that square is stored in.
                -- This logic implements that.
                0 when "0000" | "0001" | "1010" | "1011",
                1 when "0100" | "0101" | "1110" | "1111",
                2 when "1000" | "1001" | "0010" | "0011",
                3 when "1100" | "1101" | "0110" | "0111",
                0 when others;

        queued(direction) <= shard_values(shards(direction));

        skip_checkers: skip_checker port map(
            mapping => neighbours(direction).mapping,
            start_mapping_index => start_mapping_index,
            skipped => skipped(direction)
        );
    end generate;

    surface_wr_en <= run or backtrack or clear_mode;
    wr_data <= run and not clear_mode;

    -- Make this a process so that we can assign to `shard_indices` multiple times and
    -- it'll just result in overwriting.
    -- (That should never happen anyway, but the synthesiser doesn't know that.)
    process(clear_index, clear_mode, neighbours, shards)
    begin
        for i in 0 to 3 loop
            shard_indices(i) <= clear_index;
        end loop;

        if clear_mode = '0' then
            for i in 0 to 3 loop
                -- Then since each row within a chunk only contains 1 square in each shard, we
                -- can just ignore the bottom 2 bits of the x coordinate and we have our index
                -- within the shard.
                shard_indices(shards(i)) <= to_integer(neighbours(i).pos.x(coord_bits - 1 downto 2) & neighbours(i).pos.y);
            end loop;
        end if;
    end process;

    process(clear_mode, shards, neighbours_valid)
    begin
        for i in 0 to 3 loop
            shard_wr_ens(i) <= clear_mode;
        end loop;

        if clear_mode = '0' then
            for i in 0 to 3 loop
                shard_wr_ens(shards(i)) <= (run or backtrack) and neighbours_valid(i);
            end loop;
        end if;
    end process;

    gen_shards: for shard in 0 to 3 generate
        net_shards: async_ram
            generic map(size => net_len / 4)
            port map(
                clk => clk,
                addr => shard_indices(shard),
                wr_data => wr_data,
                wr_en => shard_wr_ens(shard),
                rd_data => shard_values(shard)
            );
    end generate;

    process(surface_rd_values)
    begin
        for i in 0 to 3 loop
            filled(i) <= '0';
            for cuboid in 0 to cuboids - 1 loop
                if surface_rd_values(cuboid)(i) = '1' then
                    filled(i) <= '1';
                end if;
            end loop;
        end loop;
    end process;

    neighbours_valid <= not queued and not filled and not skipped;

    gen_cuboids: for cuboid in 0 to cuboids - 1 generate
        surface_rw_indices(cuboid) <=
            clear_index when clear_mode = '1' and clear_index < area
            else to_integer(instruction.mapping(cuboid)(cursor_bits - 1 downto 2));
        gen_inner: for direction in 0 to 3 generate
            surface_rd_indices(cuboid)(direction) <= to_integer(neighbours(direction).mapping(cuboid)(cursor_bits - 1 downto 2));
        end generate;
        surfaces: surface port map(
            clk => clk,
            rw_addr => surface_rw_indices(cuboid),
            -- This'll set random extra bits to 0 when clear_mode = 1 and
            -- clear_index >= area, but that's fine: our goal is to set everything to 0
            -- anyway, it doesn't really matter if we accidentally do some of them twice.
            rw_wr_data => wr_data,
            rw_wr_en => surface_wr_en,
            rw_rd_data => surface_rw_values(cuboid),
            rd_addrs => surface_rd_indices(cuboid),
            rd_data => surface_rd_values(cuboid)
        );
    end generate;

    process(surface_rw_values)
    begin
        instruction_valid <= '1';
        for cuboid in 0 to cuboids - 1 loop
            if surface_rw_values(cuboid) = '1' then
                instruction_valid <= '0';
            end if;
        end loop;
    end process;
end arch;
