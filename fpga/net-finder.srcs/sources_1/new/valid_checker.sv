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
    output logic [3:0] neighbours_valid,
    // Whether `instruction` itself is valid.
    //
    // This only checks whether it tries to set any squares that are already
    // filled. The instruction was valid back when we were processing its
    // parent, which means it already has first dibs on its net position and
    // whether it's skipped never changes.
    output logic instruction_valid,

    // Whether each of `instruction`'s cursors are filled on the surfaces. Needed
    // for `CHECK` state.
    output logic [CUBOIDS-1:0] instruction_cursors_filled,

    // 7 series distributed RAM doesn't have any way of clearing the whole thing at
    // once, so we have to go through and individually set each bit to 0.
    input logic clear_mode,
    input logic [2*COORD_BITS-3:0] clear_index
);
  // For each neighbour of `instruction`, whether there's already a queued
  // instruction setting the same net position.
  logic [3:0] queued;
  // Whether each neighbour of `instruction` tries to set any squares on the
  // surface that are already filled.
  logic [3:0] filled;
  // Whether each neighbour of `instruction` is skipped due to being covered by
  // another finder.
  logic [3:0] skipped;

  // Find out what the neighbours of `instruction` in each direction are.
  instruction_t neighbours[4];
  always_comb begin
    for (int direction = 0; direction < 4; direction++) begin
      neighbours[direction] = instruction_neighbour(instruction, direction[1:0]);
    end
  end

  // verilator lint_off ASSIGNIN
  // verilator lint_off PINMISSING
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

    // Then instantiate all the shards. We need to put their outputs into a variable
    // like this because arrays of module instances are weird.
    logic shard_values[4];
    for (shard = 0; shard < 4; shard++) begin : gen_net_shards
      // First make a one-hot vector, where each bit just represents whether the
      // neighbour in that direction has its bit stored in this shard.
      logic [3:0] neighbour_onehot;
      for (direction = 0; direction < 4; direction++) begin : gen_neighbour_onehot
        assign neighbour_onehot[direction] = shards[direction] == shard;
      end

      // Then use a binary encoder to turn that into the actual index of the neighbour
      // using this shard.
      logic [1:0] neighbour;
      always_comb begin
        case (neighbour_onehot)
          'b0001:  neighbour = 0;
          'b0010:  neighbour = 1;
          'b0100:  neighbour = 2;
          'b1000:  neighbour = 3;
          default: neighbour = 'x;
        endcase
      end

      // Instantiate the actual RAM for this net shard.
      async_ram #(
          .SIZE(NET_LEN / 4)
      ) ram (
          .clk(clk),

          .addr(clear_mode ? clear_index : {
            neighbours[neighbour].pos.x[COORD_BITS-1:2], neighbours[neighbour].pos.y
          }),
          .wr_en(clear_mode ? 1 : (run | backtrack) & neighbours_valid[neighbour]),
          .wr_data(clear_mode ? 0 : run),
          .rd_data(shard_values[shard])
      );
    end

    // Finally, whether each neighbour's queued is whether its bit is set.
    for (direction = 0; direction < 4; direction++) begin : gen_queued
      assign queued[direction] = shard_values[shards[direction]];
    end
  endgenerate

  generate
    // Instantiate and wire up all the surfaces.
    // We need these variables to hold the outputs of the surfaces again, for
    // the same reason as before (module instances are weird).
    logic [CUBOIDS-1:0] neighbour_cursors_filled[4];
    for (cuboid = 0; cuboid < CUBOIDS; cuboid++) begin : gen_surfaces
      surface ram (
          .clk(clk),

          .rw_addr(clear_mode ? square_t'(clear_index) : instruction.mapping[cuboid].square),
          .rw_wr_en(clear_mode ? int'(clear_index) < AREA : run | backtrack),
          // We want to set this bit to 1 when running and 0 when backtracking.
          .rw_wr_data(clear_mode ? 0 : run),
          .rw_rd_data(instruction_cursors_filled[cuboid])
      );

      for (direction = 0; direction < 4; direction++) begin : gen_neighbour_cursors_filled
        assign ram.rd_addrs[direction] = neighbours[direction].mapping[cuboid].square;
        assign neighbour_cursors_filled[direction][cuboid] = ram.rd_data[direction];
      end
    end

    // Then check if `instruction` and its neighbours set any squares that area
    // already filled. Mark `instruction` as invalid if it does so.
    assign instruction_valid = !(|instruction_cursors_filled);
    for (direction = 0; direction < 4; direction++) begin : gen_filled
      assign filled[direction] = |neighbour_cursors_filled[direction];
    end
  endgenerate

  generate
    for (direction = 0; direction < 4; direction++) begin : gen_skipped
      mapping_index_t index;
      always_comb begin
        if (mapping_index_lookup(neighbours[direction].mapping, index)) begin
          skipped[direction] = index < start_mapping_index;
        end else begin
          skipped[direction] = 0;
        end
      end
    end
  endgenerate
  // verilator lint_on ASSIGNIN
  // verilator lint_on PINMISSING

  assign neighbours_valid = ~queued & ~filled & ~skipped;
endmodule
