`include "instruction_neighbour.sv"

// A single-port, 1-bit wide RAM with asynchronous read and synchronous write.
module async_ram #(
    int SIZE
) (
    input logic clk,
    input logic [$clog2(SIZE)-1:0] addr,
    input logic wr_data,
    input logic wr_en,
    output logic rd_data
);
  logic memory[SIZE];
  assign rd_data = memory[addr];
  always_ff @(posedge clk) begin
    if (wr_en) begin
      memory[addr] <= wr_data;
    end
  end
endmodule

// A RAM containing 1 bit for every square on a cuboid's surface.
//
// It has 1 port supporting both asynchronous reads and synchronous writes, and
// 4 ports that only support asynchronous reads.
module surface (
    input logic clk,

    // The read/write port.
    input square_t rw_addr,
    input logic rw_wr_data,
    input logic rw_wr_en,
    output logic rw_rd_data,

    // The read-only ports.
    input square_t rd_addrs[4],
    output logic rd_data[4]
);
  logic memory[AREA];

  assign rw_rd_data = memory[rw_addr];
  always_comb begin
    for (int i = 0; i < 4; i++) begin
      rd_data[i] = memory[rd_addrs[i]];
    end
  end

  always_ff @(posedge clk) begin
    if (rw_wr_en) begin
      memory[rw_addr] <= rw_wr_data;
    end
  end
endmodule

// An entity which checks which of an instruction's neighbours are valid, and
// also maintains all the state necessary for doing so.
module valid_checker (
    input logic clk,

    // The instruction to check the neighbours of.
    input instruction_t   instruction,
    input mapping_index_t start_mapping_index,

    // Whether `instruction` is being run and this `valid_checker`'s state needs to
    // be updated accordingly.
    input logic run,
    // Whether `instruction` is being backtracked and this `valid_checker`'s state
    // needs to be updated accordingly.
    input logic backtrack,

    // Whether `instruction`'s neighbour in each direction is valid.
    output logic neighbours_valid [4],
    // Whether `instruction` itself is valid.
    //
    // This only checks whether it tries to set any squares that are already
    // filled. The instruction was valid back when we were processing its
    // parent, which means it already has first dibs on its net position and
    // whether it's skipped never changes.
    output logic instruction_valid,

    // 7 series distributed RAM doesn't have any way of clearing the whole thing at
    // once, so we have to go through and individually set each bit to 0.
    input logic clear_mode,
    input logic [2*COORD_BITS-3:0] clear_index
);
  // For each neighbour of `instruction`, whether there's already a queued
  // instruction setting the same net position.
  logic queued [4];
  // Whether each neighbour of `instruction` tries to set any squares on the
  // surface that are already filled.
  logic filled [4];
  // Whether each neighbour of `instruction` is skipped due to being covered by
  // another finder.
  logic skipped[4];

  // Instatiate the net. It's split into 4 shards, in order to allow for 4 write
  // ports: one for each neighbour of `instruction`.
  // verilator lint_off PINMISSING
  async_ram #(.SIZE(NET_LEN / 4)) net_shards[4] (.clk(clk));

  // Find out what the neighbours of `instruction` in each direction are.
  instruction_t neighbours[4];
  always_comb begin
    for (int direction = 0; direction < 4; direction++) begin
      neighbours[direction] = instruction_neighbour(instruction, direction[1:0]);
    end
  end

  // verilator lint_off ASSIGNIN
  // Check whether each neighbour is already queued by seeing if its bit is set on the net.
  genvar direction, shard, cuboid;
  generate
    // Determine which shard of the net each neighbour's bit is located in.
    logic [1:0] shards[4];
    for (direction = 0; direction < 4; direction++) begin : gen_shards
      always_comb
        case ({
          neighbours[direction].pos.x[1:0], neighbours[direction].pos.y[1:0]
        })
          // Each 4x4 chunk of the net is split up into shards like this:
          //     2301
          //     2301
          //     0123
          //     0123
          // where each number is the shard that that square is stored in. This layout
          // guarantees that for every square, all its neighbours are in different net
          // shards.
          'b0000, 'b0001, 'b1010, 'b1011: shards[direction] = 0;
          'b0100, 'b0101, 'b1110, 'b1111: shards[direction] = 1;
          'b1000, 'b1001, 'b0010, 'b0011: shards[direction] = 2;
          'b1100, 'b1101, 'b0110, 'b0111: shards[direction] = 3;
          default: shards[direction] = 'x;
        endcase
    end

    // Then each shard's address needs to be the position of the neighbour whose bit
    // is stored there.
    for (shard = 0; shard < 4; shard++) begin : gen_net_shards
      // First make a one-hot vector, where each bit just represents whether the
      // neighbour in that direction has its bit stored in this shard.
      logic [3:0] neighbour_onehot;
      for (direction = 0; direction < 4; direction++) begin : gen_neighbour_onehot
        assign neighbour_onehot[direction] = shards[direction] == shard;
      end

      // Then use a binary encoder to turn that into the actual index of the neighbour
      // using this shard.
      always_comb begin
        logic [1:0] neighbour;
        case (neighbour_onehot)
          'b0001:  neighbour = 0;
          'b0010:  neighbour = 1;
          'b0100:  neighbour = 2;
          'b1000:  neighbour = 3;
          default: neighbour = 'x;
        endcase

        if (clear_mode) begin
          net_shards[shard].addr = clear_index;
          net_shards[shard].wr_en = 1;
          net_shards[shard].wr_data = 1;
        end else begin
          // Since each row within a chunk only contains 1 square in each shard, we can
          // just ignore the bottom 2 bits of the x coordinate and we have our index
          // within the shard.
          net_shards[shard].addr = {
            neighbours[neighbour].pos.x[COORD_BITS-1:2], neighbours[neighbour].pos.y
          };
          // We only want to write to the neighbour's bit if it's valid, otherwise this
          // neighbour wasn't queued. So when running we don't want to mark it as such,
          // and when backtracking we don't want to accidently unset this bit which was
          // set for an earlier instruction with the same position.
          net_shards[shard].wr_en = clear_mode | ((run | backtrack) & neighbours_valid[neighbour]);
          // We want to set this bit to 1 when running and 0 when backtracking.
          net_shards[shard].wr_data = run;
        end
      end
    end

    // Finally, whether each neighbour's queued is whether its bit is set.
    logic shard_values[4];
    for (shard = 0; shard < 4; shard++) assign shard_values[shard] = net_shards[shard].rd_data;
    for (direction = 0; direction < 4; direction++) begin : gen_queued
      assign queued[direction] = shard_values[shards[direction]];
    end
  endgenerate

  // Instantiate the surfaces.
  surface surfaces[4] (.clk(clk));

  generate
    // Mark `instruction` as invalid if any of the squares it sets are
    // already filled.
    // Also wire up writes to occur to the surfaces when running/backtracking.
    logic instruction_cursors_filled[CUBOIDS];
    for (cuboid = 0; cuboid < CUBOIDS; cuboid++) begin : gen_surface_rw
      always_comb begin
        if (clear_mode) begin
          surfaces[cuboid].rw_addr = square_t'(clear_index);
          surfaces[cuboid].rw_wr_en = int'(clear_index) < AREA;
          surfaces[cuboid].rw_wr_data = 0;
        end else begin
          surfaces[cuboid].rw_addr = instruction.mapping[cuboid].square;
          surfaces[cuboid].rw_wr_en = run | backtrack;
          // We want to set this bit to 1 when running and 0 when backtracking.
          surfaces[cuboid].rw_wr_data = run;
        end
      end
      assign instruction_cursors_filled[cuboid] = surfaces[cuboid].rw_rd_data;
    end
    assign instruction_valid = !|instruction_cursors_filled;

    // Mark the neighbours invalid if any of the squares they set are already filled.
    for (direction = 0; direction < 4; direction++) begin : gen_filled
      // Figure out whether each cursor that this neighbour sets is filled on its surface.
      logic neighbour_cursors_filled[CUBOIDS];
      for (cuboid = 0; cuboid < CUBOIDS; cuboid++) begin : gen_surface_rd
        assign surfaces[cuboid].rd_addrs[direction] = neighbours[direction].mapping[cuboid].square;
        assign neighbour_cursors_filled[cuboid] = surfaces[cuboid].rd_data[direction];
      end
      assign filled[direction] = |neighbour_cursors_filled;
    end
  endgenerate

  mapping_index_lookup mapping_index_lookups[4];
  generate
    for (direction = 0; direction < 4; direction++) begin : gen_skipped
      assign mapping_index_lookups[direction].mapping = neighbours[direction].mapping;
      assign skipped[direction] = mapping_index_lookups[direction].uses_fixed_class
        & mapping_index_lookups[direction].index < start_mapping_index;
    end
  endgenerate
  // verilator lint_on ASSIGNIN
  // verilator lint_on PINMISSING

  generate
    for (direction = 0; direction < 4; direction++) begin : gen_neighbours_valid
      assign neighbours_valid[direction] =
        !queued[direction] & !filled[direction] & !skipped[direction];
    end
  endgenerate
endmodule
