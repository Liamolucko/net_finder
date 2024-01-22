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
    // What `instruction`'s (probably) going to be next clock cycle, so that part of
    // next clock cycle's work can be precomputed using it.
    //
    // If this turns out to be wrong, `run` and `backtrack` must not be asserted on
    // the next clock cycle.
    input instruction_t   next_instruction,
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

    // When backtracking, which neighbours of `instruction` were valid at the time
    // it was run.
    //
    // In other words, which ones we have to unset the bits for on the net.
    input logic [3:0] neighbours_valid_in,

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
  instruction_t next_neighbours[4];

  always_ff @(posedge clk) neighbours <= next_neighbours;

  always_comb begin
    for (int direction = 0; direction < 4; direction++) begin
      next_neighbours[direction] = instruction_neighbour(next_instruction, direction[1:0]);
    end
  end

  // Check whether each neighbour is already queued by seeing if its bit is set on the net.
  genvar direction, shard, cuboid;
  generate
    // Determine which shard of the net each neighbour's bit is located in.
    logic [1:0] shards[4];
    always_ff @(posedge clk) begin
      case ({
        next_instruction.pos.x[1:0], next_instruction.pos.y[1:0]
      })
        // Each 4x4 chunk of the net is split up into shards like this:
        //     2301
        //     2301
        //     0123
        //     0123
        // where each number is the shard that that square is stored in. This layout
        // guarantees that for every square, all its neighbours are in different net
        // shards.
        'b0000, 'b1010: shards <= '{3, 0, 1, 2};
        'b0100, 'b1110: shards <= '{0, 1, 2, 3};
        'b1000, 'b0010: shards <= '{1, 2, 3, 0};
        'b1100, 'b0110: shards <= '{2, 3, 0, 1};
        'b0001, 'b1011: shards <= '{3, 2, 1, 0};
        'b0101, 'b1111: shards <= '{0, 3, 2, 1};
        'b1001, 'b0011: shards <= '{1, 0, 3, 2};
        'b1101, 'b0111: shards <= '{2, 1, 0, 3};
        default: shards <= '{default: 'x};
      endcase
    end

    // Also figure out the reverse, which neighbour falls into each shard.
    logic [1:0] shard_neighbours[4];
    logic [1:0] next_shard_neighbours[4];
    always_comb begin
      case ({
        next_instruction.pos.x[1:0], next_instruction.pos.y[1:0]
      })
        'b0000, 'b1010: next_shard_neighbours = '{1, 2, 3, 0};
        'b0100, 'b1110: next_shard_neighbours = '{0, 1, 2, 3};
        'b1000, 'b0010: next_shard_neighbours = '{3, 0, 1, 2};
        'b1100, 'b0110: next_shard_neighbours = '{2, 3, 0, 1};
        'b0001, 'b1011: next_shard_neighbours = '{3, 2, 1, 0};
        'b0101, 'b1111: next_shard_neighbours = '{0, 3, 2, 1};
        'b1001, 'b0011: next_shard_neighbours = '{1, 0, 3, 2};
        'b1101, 'b0111: next_shard_neighbours = '{2, 1, 0, 3};
        default: next_shard_neighbours = '{default: 'x};
      endcase
    end
    always_ff @(posedge clk) shard_neighbours <= next_shard_neighbours;

    // Then instantiate all the shards. We need to put their outputs into a variable
    // like this because arrays of module instances are weird.
    logic shard_values[4];
    for (shard = 0; shard < 4; shard++) begin : gen_net_shards
      logic [1:0] neighbour_direction;
      instruction_t neighbour;

      assign neighbour_direction = shard_neighbours[shard];
      always_ff @(posedge clk) neighbour <= next_neighbours[next_shard_neighbours[shard]];

      // Instantiate the actual RAM for this net shard.
      async_ram #(
          .SIZE(NET_LEN / 4)
      ) ram (
          .clk(clk),

          .addr(clear_mode ? clear_index : {neighbour.pos.x[COORD_BITS-1:2], neighbour.pos.y}),
          .wr_en(
            clear_mode ? 1
            : run ? neighbours_valid[neighbour_direction]
            : backtrack ? neighbours_valid_in[neighbour_direction]
            : 0
          ),
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
      square_t rd_addrs[4];
      surface ram (
          .clk(clk),

          .rw_addr(clear_mode ? square_t'(clear_index) : instruction.mapping[cuboid].square),
          .rw_wr_en(clear_mode ? int'(clear_index) < AREA : run | backtrack),
          // We want to set this bit to 1 when running and 0 when backtracking.
          .rw_wr_data(clear_mode ? 0 : run),
          .rw_rd_data(instruction_cursors_filled[cuboid]),

          .rd_addrs(rd_addrs),
          .rd_data ()
      );

      for (direction = 0; direction < 4; direction++) begin : gen_neighbour_cursors_filled
        assign rd_addrs[direction] = neighbours[direction].mapping[cuboid].square;
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
      mapping_index_lookup lookup (
          .mapping(neighbours[direction].mapping),
          .index(),
          .uses_fixed_class()
      );
      always_comb begin
        if (lookup.uses_fixed_class) begin
          skipped[direction] = lookup.index < start_mapping_index;
        end else begin
          skipped[direction] = 0;
        end
      end
    end
  endgenerate

  assign neighbours_valid = ~queued & ~filled & ~skipped;
endmodule
