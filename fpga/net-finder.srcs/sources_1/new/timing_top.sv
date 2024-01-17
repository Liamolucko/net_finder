`include "core.sv"

module dff (
    input  logic clk,
    input  logic d,
    output logic q
);
  always_ff @(posedge clk) begin
    q <= d;
  end
endmodule

// The top module we use in Vivado for testing the max. clock speed of `core`.
//
// It's just a simple wrapper around `core` that puts flip-flops between most of
// its inputs & outputs and the outside world. That way I/O stuff doesn't get
// included in the timing analysis.
module timing_top (
    input  logic clk,
    input  logic reset,
    input  logic in_data,
    input  logic in_valid,
    output logic in_ready,
    input  logic in_last,
    output logic out_data,
    output logic out_valid,
    input  logic out_ready,
    output logic out_last,
    output logic out_solution,
    output logic out_split,
    output logic out_pause,
    input  logic req_pause,
    input  logic req_split,
    output logic stepping
);
  // verilator lint_off ASSIGNIN
  // verilator lint_off PINMISSING
  core c (
      .clk  (clk),
      .reset(reset)
  );

  dff in_data_ff (
      .clk(clk),
      .d  (in_data),
      .q  (c.in_data)
  );
  dff in_valid_ff (
      .clk(clk),
      .d  (in_valid),
      .q  (c.in_valid)
  );
  dff in_ready_ff (
      .clk(clk),
      .d  (c.in_ready),
      .q  (in_ready)
  );
  dff in_last_ff (
      .clk(clk),
      .d  (in_last),
      .q  (c.in_last)
  );
  dff out_data_ff (
      .clk(clk),
      .d  (c.out_data),
      .q  (out_data)
  );
  dff out_valid_ff (
      .clk(clk),
      .d  (c.out_valid),
      .q  (out_valid)
  );
  dff out_ready_ff (
      .clk(clk),
      .d  (out_ready),
      .q  (c.out_ready)
  );
  dff out_last_ff (
      .clk(clk),
      .d  (c.out_last),
      .q  (out_last)
  );
  dff out_solution_ff (
      .clk(clk),
      .d  (c.out_solution),
      .q  (out_solution)
  );
  dff out_split_ff (
      .clk(clk),
      .d  (c.out_split),
      .q  (out_split)
  );
  dff out_pause_ff (
      .clk(clk),
      .d  (c.out_pause),
      .q  (out_pause)
  );
  dff req_pause_ff (
      .clk(clk),
      .d  (req_pause),
      .q  (c.req_pause)
  );
  dff req_split_ff (
      .clk(clk),
      .d  (req_split),
      .q  (c.req_split)
  );
  dff stepping_ff (
      .clk(clk),
      .d  (c.stepping),
      .q  (stepping)
  );
  // verilator lint_on ASSIGNIN
  // verilator lint_on PINMISSING
endmodule
