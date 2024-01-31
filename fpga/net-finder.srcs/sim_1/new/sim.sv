`timescale 1ns / 1ps

module sim;
  logic clk = 1;
  logic reset = 0;
  logic in_data = 0;
  logic in_valid = 0;
  logic in_last = 0;

  always begin
    #10 clk = !clk;
  end

  initial begin
    #20 reset = 1;
    #20 reset = 0;
  end

  core c (
      .clk(clk),
      .reset(reset),
      .in_data(in_data),
      .in_valid(in_valid),
      .in_ready(),
      .in_last(in_last),
      .out_data(),
      .out_valid(),
      .out_ready(1),
      .out_last(),
      .out_solution(),
      .out_split(),
      .out_pause(),
      .req_pause(0),
      .req_split(0),
      .stepping()
  );

  initial begin
    logic [25:0] finder = 26'b00010000000100000000000011;

    $dumpfile("sim.fst");
    $dumpvars();

    #70;
    for (int i = 25; i >= 0; i--) begin
      in_data  = finder[i];
      in_valid = 1;
      in_last  = i == 0;
      while (c.in_ready == 0) #20;
      #20;
      in_valid = 0;
    end

    #200000;
    $finish;
  end
endmodule
