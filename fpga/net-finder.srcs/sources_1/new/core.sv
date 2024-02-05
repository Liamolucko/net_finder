`include "valid_checker.sv"

typedef logic [$clog2(4 * AREA)-1:0] potential_index_t;
typedef logic [$clog2(4 * AREA)-1:0] decision_index_t;
// Make `AREA` a valid `run_stack_index_t` so that we can use
// `next_target_ref.parent >= next_run_stack_len` to check if we're about to go
// past the end of the queue without worrying about overflow.
typedef logic [$clog2(AREA+1)-1:0] run_stack_index_t;

typedef struct packed {
  // The index of the instruction's parent in the run stack.
  run_stack_index_t parent;
  // The index of this instruction in its parent's list of valid children.
  //
  // If the index is past the end of that list, it represents the last valid
  // child. Then we always store the last valid child as 11, so that when
  // backtracking we can immediately see 'oh this is the last one, so we need to
  // move onto the next instruction'.
  logic [1:0] child_index;
} instruction_ref_t;

typedef struct packed {
  // The instruction that was run.
  instruction_t instruction;
  // A reference to where in the run stack this instruction originally came from.
  instruction_ref_t instruction_ref;
  // Whether this instruction's child in each direction was valid at the time this
  // instruction was run.
  logic [3:0] children;
  // The number of potential instructions there were at the point when it was run.
  potential_index_t potential_len;
  // The index of the decision to run this instruction in the list of decisions.
  decision_index_t decision_index;
} run_stack_entry_t;

// This should really be an enum, but my FSM is structured too weirdly for
// Vivado to detect it and so I have to manually convert it to be one-hot.
typedef struct packed {
  // The core is currently clearing out all its internal memory after a reset.
  logic clear;
  // The core is currently receiving or waiting to receive a finder to run, and
  // filling in its state.
  logic receive;
  // The core is actively attempting to run instructions.
  logic run;
  // The core is backtracking the last instruction in the run stack.
  //
  // This only just barely requires it's own state: the other way to do this would
  // be to fold it into `RUN`, and backtrack on the same clock cycle that you find
  // out that the last instruction in the queue produces no new instructions for
  // you to try and run.
  //
  // However, that would actually end up needing two 4-way neighbour lookups,
  // since you need to first look up the neighbours of the last instruction in the
  // queue in order to check if any of them are valid (and thus whether that last
  // instruction is a potential instruction), and then look up the neighbours of
  // the instruction to backtrack so their bits can be unset on the net.
  logic backtrack;
  // The core is waiting for its leftover state from the last `CHECK` to be
  // cleared out before switching into `CHECK` state.
  logic check_wait;
  // The core is checking if its current solution is valid.
  logic check;
  // The core is outputting a potential solution.
  logic solution;
  // The core incorrectly predicted what the next `target_ref` would be and has to
  // correct the value of `target`.
  logic stall;
  // The core has split its finder in half and is currently outputting one of
  // those halves.
  logic split;
  // The core is being paused; it's currently writing out the state of its finder.
  // Once it's done, it'll switch to `receive` state.
  logic pause;
} state_t;

parameter state_t CLEAR = '{clear: 1, default: 0};
parameter state_t RECEIVE = '{receive: 1, default: 0};
parameter state_t RUN = '{run: 1, default: 0};
parameter state_t BACKTRACK = '{backtrack: 1, default: 0};
parameter state_t CHECK_WAIT = '{check_wait: 1, default: 0};
parameter state_t CHECK = '{check: 1, default: 0};
parameter state_t SOLUTION = '{solution: 1, default: 0};
parameter state_t STALL = '{stall: 1, default: 0};
parameter state_t SPLIT = '{split: 1, default: 0};
parameter state_t PAUSE = '{pause: 1, default: 0};

// All the metadata about a finder that gets sent prior to the list of
// decisions.
typedef struct packed {
  // The mapping of the starting instruction.
  mapping_t start_mapping;
  // The index of that mapping.
  mapping_index_t start_mapping_index;
  // The index in the list of decisions of the first decision it's this finder's
  // job to try both options of.
  //
  // The decision at this index is always a 1, since when `base_decision` points
  // to a 0 it's the same as if it pointed to the next 1 - you can't backtrack the
  // 0 anyway so what's the point of saying you're allowed to?
  //
  // Unlike the rest of the data here this can actually be mutated over time (when
  // splitting and when the decision at this index becomes a 0).
  decision_index_t base_decision;
} prefix_t;

// A simple-dual-port asynchronous RAM for storing run stacks.
module run_stack_ram (
    input logic clk,

    input run_stack_index_t wr_addr,
    input run_stack_entry_t wr_data,
    input logic wr_en,

    input  run_stack_index_t rd_addr,
    output run_stack_entry_t rd_data
);
  logic [$bits(run_stack_entry_t)-1:0] memory[AREA];

  // It doesn't matter that we might miss a write that occurs on the same clock
  // cycle that we set `target`, because we have a check for it and STALL if it
  // happens.
  assign rd_data = memory[rd_addr];
  always_ff @(posedge clk) begin
    if (wr_en) begin
      memory[wr_addr] <= wr_data;
    end
  end
endmodule

module core (
    // The clock signal for the core.
    input logic clk,
    // A synchronous reset signal for the core.
    input logic reset,

    // This is loosely inspired by AXI4-Stream, except that it operates on bits
    // instead of bytes.
    //
    // `in_valid` is set whenever `in_data` is actually valid data, and
    // `in_ready` is set whenever the core is ready to accept data. When
    // both of them are set, a bit is received.
    //
    // `in_last` is whether this is the last bit of the finder's details.
    //
    // As for what we actually send through here, it's finders for this core
    // to run. They're formatted as:
    // - The mapping the finder's first instruction maps (0, 0) on the net to.
    //   - We internally use this to calculate `start_mapping_index` as well.
    // - Then, for every instruction that was valid (didn't try to set the same net
    //   position as another instruction, didn't try to set a spot on the surface
    //   that was already filled, and wasn't skipped), whether or not that
    //   instruction was run.
    input  logic in_data,
    input  logic in_valid,
    output logic in_ready,
    input  logic in_last,

    // These signals work the same way as `in_*` except this time the core
    // is the one sending the data.
    output logic out_data,
    output logic out_valid,
    input  logic out_ready,
    output logic out_last,

    // Whether the finder being emitted is a solution.
    output logic out_solution,
    // Whether the finder being emitted is a response to `req_split`.
    output logic out_split,
    // Whether the finder being emitted is a response to `req_pause`.
    output logic out_pause,

    // These can be set to 1 to request that this core pause or split.
    // Pausing takes priority.
    input logic req_pause,
    input logic req_split,

    // Whether this core needs a new finder to run.
    //
    // This is asserted when the core enters RECEIVE state, and deasserted as soon
    // as it starts receiving a finder.
    output logic wants_finder,

    // Whether the next clock cycle will be put to use for either running or
    // backtracking. For testing purposes.
    output logic   stepping,
    // The current state the core's in. For profiling and/or debugging purposes.
    output state_t state
);

  state_t state_reg;
  state_t next_state;
  state_t predicted_state;

  always_ff @(posedge clk) begin
    if (reset | local_reset) begin
      state_reg <= CLEAR;
    end else begin
      state_reg <= next_state;
    end
  end

  assign state = state_reg;

  assign out_solution = out_valid & state.solution;
  assign out_split = out_valid & state.split;
  assign out_pause = out_valid & state.pause;

  // Control signals.
  // Whether to do a synchronous reset on the next clock edge, independent of the
  // rest of the design.
  logic local_reset;
  // Whether to advance to the next instruction.
  logic advance;
  // Whether to run `target`. Should only ever be 1 if `advance` is 1 as well.
  logic run;
  // Whether to backtrack the last instruction in the run stack (which must be
  // `target_parent`).
  logic backtrack;
  // Whether to add 1 to `decision_index`.
  logic inc_decision_index;
  // Whether we're currently sending `prefix`/`decisions`, as opposed to receiving
  // into them.
  logic sending;
  // Whether to reset `prefix_bits` back to `$bits(prefix_t)`.
  logic reset_prefix_bits;

  assign stepping = run | backtrack;

  // Collect that metadata into a shift register.
  prefix_t prefix;
  prefix_t next_prefix;

  // How many more bits `prefix` needs to be shifted - either in order to finish
  // sreading it in or writing it out.
  logic [$clog2($bits(prefix)+1)-1:0] prefix_bits_left;
  logic [$clog2($bits(prefix)+1)-1:0] next_prefix_bits_left;

  // The bit currently being shifted out of `prefix`.
  logic prefix_out;
  assign prefix_out = prefix[$bits(prefix)-1];

  decision_index_t base_decision;
  assign base_decision = prefix.base_decision;
  // What we're going to set `prefix.base_decision` to next clock cycle (if
  // `prefix` isn't currently halfway through being shifted in/out of the core).
  decision_index_t next_base_decision;

  // We don't bother clearing `prefix` on reset since it's bogus anyway.
  always_ff @(posedge clk) prefix <= next_prefix;

  always_ff @(posedge clk) begin
    if (reset | local_reset | reset_prefix_bits) begin
      prefix_bits_left <= $bits(prefix_t);
    end else begin
      prefix_bits_left <= next_prefix_bits_left;
    end
  end

  always_comb begin
    next_prefix = prefix;
    next_prefix_bits_left = prefix_bits_left;
    if (prefix_bits_left > 0 & ((state.receive & in_valid) | (sending & out_ready))) begin
      next_prefix = {prefix[$bits(prefix)-2:0], sending ? prefix[$bits(prefix)-1] : in_data};
      next_prefix_bits_left = prefix_bits_left - 1;
    end else if (prefix_bits_left == 0) begin
      next_prefix.base_decision = next_base_decision;
    end
  end

  // Recreating the list of decisions this core made (whether to run/not run every
  // valid instruction it came across) is surprisingly difficult; it requires
  // making a whole second copy of the core's state.
  // So instead, we just store it as we go along.
  async_ram #(
      .SIZE(4 * AREA)
  ) decisions (
      .clk(clk),

      .addr(decisions_addr),
      .wr_en(decisions_wr_en),
      .wr_data(decisions_wr_data),
      .rd_data(decisions_rd_data)
  );

  // The number of decisions we've made.
  logic [$clog2(4*AREA+1)-1:0] decisions_len;
  logic [$clog2(4*AREA+1)-1:0] next_decisions_len;

  // stupid broken-out signals for the ports of `decisions`
  decision_index_t decisions_addr;
  logic decisions_wr_en;
  logic decisions_wr_data;
  logic decisions_rd_data;

  always_ff @(posedge clk) begin
    if (reset | local_reset) begin
      decisions_len <= 0;
    end else begin
      decisions_len <= next_decisions_len;
    end
  end

  always_comb begin
    automatic decision_index_t last_run = 'x;

    if (state.solution | state.split | state.pause) begin
      // We're reading from the current index.
      decisions_addr = decision_index;
    end else if (state.backtrack) begin
      last_run = target_parent.decision_index;
      // We're changing the last 1 in `decision` to a 0.
      decisions_addr = last_run;
    end else begin
      // We're adding a 1 to the end of `decisions`.
      decisions_addr = decisions_len;
    end
  end

  // The index of the current decision we're sending.
  decision_index_t decision_index;
  always_ff @(posedge clk) begin
    if (reset | local_reset | (!state.solution & !state.split & !state.pause)) begin
      // Set this to 0 whenever we aren't using it so that way it'll always be 0 when
      // we start to.
      decision_index <= 0;
    end else if (inc_decision_index) begin
      decision_index <= decision_index + 1;
    end
  end

  // The list of instructions that have been run.
  run_stack_ram run_stack (
      .clk(clk),
      .wr_addr(run_stack_len),
      .wr_en(run),
      .wr_data(
      '{
          instruction: target,
          // This is nonsense for the first instruction.
          instruction_ref: '{
              parent: target_ref.parent,
              child_index: last_child ? 3 : target_ref.child_index
          },
          children: vc.neighbours_valid,
          potential_len: potential_len,
          decision_index: decisions_len
      }
      ),
      .rd_addr(predicted_target_ref.parent),
      .rd_data(next_target_parent)
  );
  logic [$clog2(AREA+1)-1:0] run_stack_len;
  logic [$clog2(AREA+1)-1:0] next_run_stack_len;

  // `run_stack_len` + 1.
  logic [$clog2(AREA+1)-1:0] run_stack_len_inc;
  // `run_stack_len` - 1.
  logic [$clog2(AREA+1)-1:0] run_stack_len_dec;

  always_ff @(posedge clk) begin
    run_stack_len_inc <= next_run_stack_len + 1;
    run_stack_len_dec <= next_run_stack_len - 1;
  end

  // A reference to the instruction we're currently looking at.
  //
  // This always lines up with `target`, except when `run_stack_len` is 0 and in
  // `CHECK` state.
  instruction_ref_t target_ref;
  // The parent of `target`'s entry in the run stack.
  run_stack_entry_t target_parent;

  // The instruction we're currently looking at.
  instruction_t target;
  // Whether `target` is the last child of `target_parent` we have to look at.
  logic last_child;

  // A reference to the instruction we're going to switch to looking at on the
  // next clock edge.
  instruction_ref_t next_target_ref;
  // The parent of `next_target`'s entry in the run stack.
  run_stack_entry_t next_target_parent;
  // The instruction we're going to switch to looking at on the next clock edge.
  instruction_t next_target;
  // Whether `next_target` is the last child of `next_target_parent` we have to
  // look at.
  logic next_last_child;

  // A guess at what `next_target_ref` will be before looking at `target`, so that
  // we can calculate next clock cycle's `target` in parallel before finding out
  // for sure what the `target_ref` will be.
  //
  // If we're wrong, a clock cycle gets wasted in STALL state.
  instruction_ref_t predicted_target_ref;

  always_ff @(posedge clk) begin
    if (reset | local_reset) begin
      run_stack_len <= 0;
      // Don't bother resetting `target_ref`, `target_parent` and `last_child`,
      // they're all nonsense until `run_stack_len` is at least 1 anyway.
      // `target` is also nonsense, but in its case until `prefix_bits_left` is 0.
    end else begin
      run_stack_len <= next_run_stack_len;
      target_ref <= next_target_ref;
      target_parent <= next_target_parent;
      target <= next_target;
      last_child <= next_last_child;
    end
  end

  assert property (@(posedge clk) next_target_parent.children == 'b0000 |->
  // `target_ref` is going to be garbage, so we need to be entering a state where
  // we don't care about that.
  next_run_stack_len == 0 | next_state.clear | next_state.check_wait | next_state.solution
    | next_state.stall | next_state.split | next_state.pause
  );
  always_comb begin
    automatic logic [1:0] last_valid_direction = 'x;
    automatic int current_child = 'x;
    // The direction of `next_target` from `next_target_parent`.
    automatic logic [1:0] next_target_direction = 'x;

    if (run_stack_len == 0 & !advance) begin
      // We must not have received any decisions yet, and still won't have next clock
      // cycle since otherwise we'd be advancing, and so our next target has the
      // mapping we've received and a default net position of (0, 0).
      next_target = '{pos: '{x: 0, y: 0}, mapping: next_prefix.start_mapping};
      // This is nonsense, since `target` is the first instruction and doesn't have a
      // parent.
      next_last_child = 'x;
    end else begin
      // Find the direction of the last valid child.
      last_valid_direction = 'x;
      for (int direction = 3; direction >= 0; direction--) begin
        if (next_target_parent.children[direction]) begin
          last_valid_direction = direction[1:0];
          break;
        end
      end

      // To start off with, set `target_direction` to the last valid direction. That
      // way, if `child_index` is past the end of the list of valid children, it
      // defaults to the last one.
      next_target_direction = last_valid_direction;

      // Then check if there's an exact match.
      current_child = -1;
      for (int direction = 0; direction < 4; direction++) begin
        if (next_target_parent.children[direction]) begin
          // The child is valid so it actually gets a new index.
          current_child += 1;
          if (current_child == int'(predicted_target_ref.child_index)) begin
            next_target_direction = direction[1:0];
            break;
          end
        end
      end

      next_target = instruction_neighbour(next_target_parent.instruction, next_target_direction);
      next_last_child = next_target_direction == last_valid_direction;
    end
  end

  // The list of potential instructions.
  // I would've just named it `potential` but that's a keyword in SystemVerilog.
  instruction_ref_t potential_buf[4 * AREA];

  // The length of `potential_buf`.
  //
  // In theory might need more bits than `potential_index_t`, except that it's
  // impossible for the length of `potential_buf` to max out like that anyway
  // since it would require the first instruction to be a potential instruction.
  // And doing it this way guarantees that this uses the same number of bits as a
  // cursor.
  potential_index_t potential_len;
  potential_index_t next_potential_len;

  always_ff @(posedge clk) begin
    if (reset | local_reset) begin
      potential_len   <= 0;
      potential_index <= 0;
    end else begin
      potential_len   <= next_potential_len;
      potential_index <= next_potential_index;
    end
  end

  always_ff @(posedge clk) begin
    if (advance & vc.instruction_valid & !(|vc.neighbours_valid)) begin
      // We're advancing, and the instruction is valid but none of its neighbours are;
      // that means this is a potential instruction. Add it to the list.
      potential_buf[potential_len] <= target_ref;
    end
  end

  // The index of the potential instruction we're currently looking at (i.e., that
  // `target` is set to if we're in `CHECK` state).
  //
  // Note that this isn't the index we're actually indexing into `potential_buf`
  // right now - that's `next_potential_index`, so that we can set
  // `next_target_ref` to that potential instruction in time to set `target_ref`
  // and `target` on the next clock edge at the same time as we set
  // `potential_index`.
  potential_index_t potential_index;
  potential_index_t next_potential_index;

  // The next potential instruction we're going to be looking at.
  instruction_ref_t next_potential_target;

  always_comb begin
    if (state.check) begin
      next_potential_index = potential_index + 1;
    end else begin
      next_potential_index = 0;
    end
  end

  assign next_potential_target = potential_buf[next_potential_index];

  // When in `CLEAR` state, the index in the net/surface we're currently clearing.
  logic [2*COORD_BITS-3:0] clear_index;
  always_ff @(posedge clk) begin
    if (reset | local_reset | !state.clear) begin
      // Reset this to 0 whenever we aren't in `CLEAR` state. That way, it'll always
      // be 0 on the first clock cycle of clearing.
      clear_index <= 0;
    end else begin
      // Then while in `CLEAR` state we increment every clock cycle.
      clear_index <= clear_index + 1;
    end
  end

  valid_checker vc (
      .clk(clk),

      .instruction(state.backtrack ? target_parent.instruction : target),
      // In the case that `backtrack` is 0 despite the state being `BACKTRACK`, it
      // won't matter that this is wrong since we'll be splitting or pausing anyway,
      // neither of which require `valid_checker`.
      //
      // In the case that `predicted_state` is wrong, it won't matter since we'll
      // STALL and fix it (even if `predicted_target_ref` happened to be correct, we
      // have a special case for this).
      .next_instruction(predicted_state.backtrack ? next_target_parent.instruction : next_target),
      .start_mapping_index(prefix.start_mapping_index),

      .run(run),
      .backtrack(backtrack),

      .neighbours_valid (),
      .instruction_valid(),

      .neighbours_valid_in(target_parent.children),

      .instruction_cursors_filled(),

      .clear_mode (state.clear),
      .clear_index(clear_index)
  );

  // Instantiate RAMs used for keeping track of which squares on the surfaces are
  // filled by potential instructions.
  //
  // Most of the time they're maintained as all 0s, and then they're filled in in
  // `CHECK` state.
  // First we declare their signals, then actually instantiate them inside a `generate`.

  // How many bits would be set on each surface if all the potential instructions
  // were run (ignoring that they might conflict).
  //
  // In other words, `potential_areas[cuboid]` is `run_stack_len` + the number of
  // 1s set in `potential_surfaces[cuboid]`.
  logic [$clog2(AREA+1)-1:0] potential_areas[CUBOIDS];
  logic [$clog2(AREA+1)-1:0] next_potential_areas[CUBOIDS];
  always_ff @(posedge clk) begin
    if (reset | local_reset) begin
      for (int cuboid = 0; cuboid < CUBOIDS; cuboid++) begin
        // We don't actually have any idea how many 1s there are at this point, but
        // since we go into `CLEAR` state on reset 0 will become correct in a second
        // anyway.
        potential_areas[cuboid] <= 0;
      end
    end else begin
      potential_areas <= next_potential_areas;
    end
  end

  // Once we exit `CHECK` state, we need to reset `potential_surfaces` to all 0s
  // ready for the next time we enter `CHECK` state. This is the index we're up to
  // in doing so.
  square_t potential_surfaces_clear_index;
  always_ff @(posedge clk) begin
    if (reset | local_reset | state.check) begin
      potential_surfaces_clear_index <= 0;
    end else begin
      potential_surfaces_clear_index <= potential_surfaces_clear_index + 1;
    end
  end

  square_t [CUBOIDS-1:0] potential_surface_addrs;
  logic [CUBOIDS-1:0] potential_surface_rd_datas;
  logic [CUBOIDS-1:0] potential_surface_wr_datas;
  logic [CUBOIDS-1:0] potential_surface_wr_ens;

  generate
    genvar cuboid;
    for (cuboid = 0; cuboid < CUBOIDS; cuboid++) begin : gen_potential_surfaces
      surface ram (
          .clk(clk),
          // This input should be computed inline, since it affects `rd_data` and so if it
          // were computed inside `comb_main` `rd_data` wouldn't be set until after the
          // end of the `always_comb`, and the copy of it that computes
          // `predicted_target_ref` would be ignored.
          //
          // It shouldn't make a difference in practice since this doesn't depend on
          // `valid_checker` and so the two calls won't differ, but I want to realise it
          // if they ever do.
          .rw_addr(
            state.clear ? square_t'(clear_index)
            : state.check ? target.mapping[cuboid].square
            : potential_surfaces_clear_index
          ),
          .rw_rd_data(potential_surface_rd_datas[cuboid]),
          .rw_wr_data(potential_surface_wr_datas[cuboid]),
          .rw_wr_en(potential_surface_wr_ens[cuboid]),

          // We don't need the read ports.
          .rd_addrs(),
          .rd_data ()
      );
    end
  endgenerate

  // When in SPLIT state, whether we were in BACKTRACK state before switching to
  // SPLIT state.
  logic was_backtrack;
  logic next_was_backtrack;
  always_ff @(posedge clk) was_backtrack <= next_was_backtrack;

  // When in STALL state, the state we need to switch to after correcting
  // `target`.
  state_t saved_state;
  state_t next_saved_state;
  always_ff @(posedge clk) saved_state <= next_saved_state;

  // Various comparisons of registers.
  // Whether or not it's valid to backtrack `target_parent`.
  logic can_backtrack;
  // Whether or not we're currently capable of splitting (in other words, whether
  // `decisions[base_decision]` exists).
  logic can_split;
  always_ff @(posedge clk) begin
    can_backtrack <= next_target_parent.decision_index >= next_base_decision;
    can_split <= next_base_decision < next_decisions_len;
  end

  // Whether we're being interrupted by `req_pause`/`req_split`.
  //
  // Note that this is 0 if `req_split` is 1 and `can_split` is 0; the only case
  // where this is 1 and `interrupted` is 0 is when `state` isn't RUN or
  // BACKTRACK.
  logic interrupting;
  // Whether we're actually going to respond to that interruption.
  logic interrupted;
  always_comb begin
    interrupting = 0;
    if (req_pause) begin
      // We always pause immediately upon request.
      interrupting = 1;
    end else if (req_split & can_split) begin
      // We only split if `base_decision` is within `decisions`, since we need to
      // split on it. If it isn't, we continue until either:
      // - An instruction is run, at which point `base_decision` now points to the
      //   decision to do so and we can split on it.
      // - We get to the end of the queue without running anything; that means
      //   `base_decision` is still 1 past the end of `decisions`, and thus there are
      //   no decisions we're allowed to backtrack and we're done.
      interrupting = 1;
    end
    interrupted = interrupting & (state.run | state.backtrack);
  end

  always_comb begin
    // Do this out here to make sure `advance`/`backtrack` don't depend on
    // `instruction_valid`/`neighbours_valid`.
    advance   = 0;
    backtrack = 0;
    case (state)
      CLEAR, CHECK_WAIT, CHECK, SOLUTION, STALL, SPLIT, PAUSE: begin
        advance   = 0;
        backtrack = 0;
      end
      RECEIVE: advance = prefix_bits_left == 0 & in_valid;
      RUN: advance = !interrupting;
      BACKTRACK: backtrack = !interrupting & can_backtrack;
      default: begin
        advance   = 'x;
        backtrack = 'x;
      end
    endcase
  end

  assign sending = state.solution | state.split | state.pause;

  // In parallel with the main `comb_main` that uses the outputs of
  // `valid_checker`, we run another one where they're just hardcoded to 0. That
  // way it can calculate `predicted_target_ref` at the same time as
  // `valid_checker` is running.
  vc_dependent #(
      .TARGET_IGNORED_STATES(0)
  ) dep_fake (
      .interrupted(interrupted),
      .in_data(in_data),
      .in_valid(in_valid),
      .in_last(in_last),
      .out_ready(out_ready),
      .state(state),
      .prefix_bits_left(prefix_bits_left),
      .prefix_out(prefix_out),
      .base_decision(base_decision),
      .decisions_len(decisions_len),
      .decisions_rd_data(decisions_rd_data),
      .decision_index(decision_index),
      .run_stack_len(run_stack_len),
      .target_ref(target_ref),
      .target_parent(target_parent),
      .last_child(last_child),
      .potential_len(potential_len),
      .clear_index(clear_index),
      .potential_surface_rd_datas(potential_surface_rd_datas),
      .potential_areas(potential_areas),
      .potential_surfaces_clear_index(potential_surfaces_clear_index),
      .was_backtrack(was_backtrack),
      .saved_state(saved_state),
      .advance(advance),
      .backtrack(backtrack),

      .run_stack_len_inc(run_stack_len_inc),
      .run_stack_len_dec(run_stack_len_dec),
      .can_backtrack(can_backtrack),

      .next_potential_index (next_potential_index),
      .next_potential_target(next_potential_target),

      .instruction_valid(0),
      .neighbours_valid (0),

      .next_state(predicted_state),
      .next_target_ref(predicted_target_ref),

      // All these outputs will be overriden by the real `comb_main` in a sec.
      .next_base_decision(),
      .decisions_wr_data(),
      .decisions_wr_en(),
      .next_decisions_len(),
      .next_run_stack_len(),
      .next_potential_len(),
      .potential_surface_wr_datas(),
      .potential_surface_wr_ens(),
      .next_potential_areas(),
      .next_saved_state(),

      .in_ready(),
      .out_data(),
      .out_valid(),
      .out_last(),
      .wants_finder(),
      .local_reset(),
      .run(),
      .inc_decision_index(),
      .reset_prefix_bits()
  );

  // Then run the one which does it properly.
  vc_dependent #(
      .TARGET_IGNORED_STATES(1)
  ) dep_real (
      .interrupted(interrupted),
      .in_data(in_data),
      .in_valid(in_valid),
      .in_last(in_last),
      .out_ready(out_ready),
      .state(state),
      .prefix_bits_left(prefix_bits_left),
      .prefix_out(prefix_out),
      .base_decision(base_decision),
      .decisions_len(decisions_len),
      .decisions_rd_data(decisions_rd_data),
      .decision_index(decision_index),
      .run_stack_len(run_stack_len),
      .target_ref(target_ref),
      .target_parent(target_parent),
      .last_child(last_child),
      .potential_len(potential_len),
      .clear_index(clear_index),
      .potential_surface_rd_datas(potential_surface_rd_datas),
      .potential_areas(potential_areas),
      .potential_surfaces_clear_index(potential_surfaces_clear_index),
      .was_backtrack(was_backtrack),
      .saved_state(saved_state),
      .advance(advance),
      .backtrack(backtrack),

      .run_stack_len_inc(run_stack_len_inc),
      .run_stack_len_dec(run_stack_len_dec),
      .can_backtrack(can_backtrack),

      .next_potential_index (next_potential_index),
      .next_potential_target(next_potential_target),

      .instruction_valid(vc.instruction_valid),
      .neighbours_valid (vc.neighbours_valid),

      // A couple of these are left out so they can be assigned then overriden inside
      // the below `always_comb`.
      .next_state(),
      .next_target_ref(next_target_ref),

      .next_base_decision(),
      .decisions_wr_data(decisions_wr_data),
      .decisions_wr_en(decisions_wr_en),
      .next_decisions_len(next_decisions_len),
      .next_run_stack_len(next_run_stack_len),
      .next_potential_len(next_potential_len),
      .potential_surface_wr_datas(potential_surface_wr_datas),
      .potential_surface_wr_ens(potential_surface_wr_ens),
      .next_potential_areas(next_potential_areas),
      .next_saved_state(),

      .in_ready(in_ready),
      .out_data(out_data),
      .out_valid(out_valid),
      .out_last(out_last),
      .wants_finder(wants_finder),
      .local_reset(local_reset),
      .run(run),
      .inc_decision_index(inc_decision_index),
      .reset_prefix_bits()
  );

  always_comb begin
    // Whether `target_parent`+`target` are going to be wrong on the next clock
    // cycle (usually because of `predicted_target_ref` being wrong).
    automatic logic invalid_target;
    // Whether that matters for the state we're transitioning into.
    automatic logic target_matters;

    next_state = dep_real.next_state;
    next_base_decision = dep_real.next_base_decision;
    next_saved_state = dep_real.next_saved_state;
    reset_prefix_bits = dep_real.reset_prefix_bits;

    next_was_backtrack = was_backtrack;
    if (interrupted) begin
      if (req_pause) begin
        reset_prefix_bits = 1;
        next_state = PAUSE;
      end else begin
        assert (req_split & base_decision < decisions_len);
        reset_prefix_bits = 1;
        // Increment `base_decision` now, since we immediately start sending `prefix`
        // and need it to be correct.
        //
        // This isn't necessarily what we actually want the new `prefix.base_decision`
        // to be, but we correct that later.
        next_base_decision = base_decision + 1;
        // Save whether we need to switch to RUN or BACKTRACK state after splitting.
        next_was_backtrack = state.backtrack;
        next_state = SPLIT;
      end
    end

    invalid_target = next_target_ref != predicted_target_ref
    // `target_parent`'s going to be invalid since the instruction it's supposed to
    // be set to only gets written to `run_stack` on the same clock edge as
    // `target_parent` is set. So, STALL.
    | (next_run_stack_len != 0 & next_target_ref.parent == run_stack_len)
    // Special-cased since `valid_checker` relies on it.
    | next_state.backtrack != predicted_state.backtrack;

    target_matters = (next_state.receive & next_prefix_bits_left == 0) | next_state.run
    | next_state.backtrack | next_state.check;

    if (invalid_target & target_matters) begin
      // Oops, `target`'s going to be incorrect now, so we have to go into `STALL`
      // state and fix it.
      next_saved_state = next_state;
      next_state = STALL;
    end
  end
endmodule

// All of our combinational logic that might depend on the outputs of `valid_checker`.
module vc_dependent #(
    // Whether `next_state`s which don't care about the value of `target` should be
    // considered.
    //
    // When this is 0, we pretend they don't happen, which is used to simplify the
    // logic of `dep_fake` without actually affecting results since what it outputs
    // only matters if we're switching into a state which cares about `target`.
    logic TARGET_IGNORED_STATES = 1
) (
    // Whether `req_pause` or `req_split` caused us to not advance/backtrack.
    input logic interrupted,

    input logic in_data,
    input logic in_valid,
    input logic in_last,
    input logic out_ready,
    input state_t state,
    input logic [$clog2($bits(prefix_t)+1)-1:0] prefix_bits_left,
    input logic prefix_out,
    input decision_index_t base_decision,
    input logic [$clog2(4*AREA+1)-1:0] decisions_len,
    input logic decisions_rd_data,
    input decision_index_t decision_index,
    input logic [$clog2(AREA+1)-1:0] run_stack_len,
    input instruction_ref_t target_ref,
    input run_stack_entry_t target_parent,
    input logic last_child,
    input potential_index_t potential_len,
    input logic [2*COORD_BITS-3:0] clear_index,
    input logic [CUBOIDS-1:0] potential_surface_rd_datas,
    input logic [$clog2(AREA+1)-1:0] potential_areas[CUBOIDS],
    input square_t potential_surfaces_clear_index,
    input logic was_backtrack,
    input state_t saved_state,
    input logic advance,
    input logic backtrack,

    // Cached stuff.
    input logic [$clog2(AREA+1)-1:0] run_stack_len_inc,
    input logic [$clog2(AREA+1)-1:0] run_stack_len_dec,
    input logic can_backtrack,

    input potential_index_t next_potential_index,
    input instruction_ref_t next_potential_target,

    input logic instruction_valid,
    input logic [3:0] neighbours_valid,

    output state_t next_state,
    output instruction_ref_t next_target_ref,

    output decision_index_t next_base_decision,
    output logic decisions_wr_data,
    output logic decisions_wr_en,
    output logic [$clog2(4*AREA+1)-1:0] next_decisions_len,
    output logic [$clog2(AREA+1)-1:0] next_run_stack_len,
    output potential_index_t next_potential_len,
    output logic [CUBOIDS-1:0] potential_surface_wr_datas,
    output logic [CUBOIDS-1:0] potential_surface_wr_ens,
    output logic [$clog2(AREA+1)-1:0] next_potential_areas[CUBOIDS],
    output state_t next_saved_state,

    output logic in_ready,
    output logic out_data,
    output logic out_valid,
    output logic out_last,
    output logic wants_finder,
    output logic local_reset,
    output logic run,
    output logic inc_decision_index,
    output logic reset_prefix_bits
);
  always_comb begin
    automatic logic clearing = 'x;
    automatic instruction_ref_t to_backtrack = 'x;
    automatic decision_index_t last_run = 'x;
    automatic logic maybe_solution = 'x;
    automatic logic [CUBOIDS-1:0] inc_potential_areas = 0;

    in_ready = 0;
    // This is almost always what it should be set to, except when we override it to 0 during splitting.
    out_data = prefix_bits_left > 0 ? prefix_out : decisions_rd_data;
    out_valid = 0;
    out_last = 0;
    wants_finder = 0;

    local_reset = 0;
    run = 0;
    inc_decision_index = 0;
    reset_prefix_bits = 0;

    // By default, stay in the current state.
    next_state = state;
    next_base_decision = base_decision;
    next_saved_state = saved_state;

    case (state)
      CLEAR: begin
        if (int'(clear_index) == NET_LEN / 4 - 1 | !TARGET_IGNORED_STATES) begin
          // We're about to clear the last bit of the net, in which case we're done clearing.
          next_state = RECEIVE;
        end
      end

      RECEIVE: begin
        wants_finder = prefix_bits_left == $bits(prefix_t);
        if (prefix_bits_left == 0) begin
          // We're up to processing decisions now.
          if (instruction_valid & |neighbours_valid) begin
            // The instruction's valid and isn't a potential instruction, which means that
            // whether we run it's a decision and we can consume one if one's ready.
            in_ready = 1;
            run = in_valid & in_data;
          end
        end else begin
          // We always immediately read the prefix.
          in_ready = 1;
          // The finder shouldn't cut off partway through receiving the prefix.
          assert (!(in_valid & in_last));
        end

        if (in_valid & in_ready & in_last) begin
          // Note: this will get overridden later if there's no next instruction to run.
          next_state = RUN;
        end
      end

      RUN: begin
        // Note: this is violated when running the first instruction, however that
        // should always happen inside RECEIVE state not RUN state.
        assert (target_ref.parent < run_stack_len);

        if (!interrupted) begin
          if (instruction_valid & |neighbours_valid) begin
            // The instruction's valid and isn't a potential instruction, run it.
            run = 1;
          end
        end

        // No, we don't just stay in RUN state forever; there's some code later which
        // overrides next_state from RUN to CHECK_WAIT/CHECK/BACKTRACK (or even issues a
        // synchronous reset) if we're at the end of the queue.
      end

      BACKTRACK: begin
        assert (target_ref.parent == run_stack_len - 1);

        if (!interrupted) begin
          if (!can_backtrack) begin
            // Hold up, we're about to try and backtrack an decision that it isn't this
            // finder's job to try both options of. That means we're done!
            local_reset = 1;
          end else begin
            next_state = RUN;

            if (target_parent.decision_index == base_decision) begin
              // The base decision is being backtracked, and so it doesn't really make sense
              // to call it the base decision anymore. What being the base decision means is
              // that it's the earliest decision that we need to backtrack; but we've just
              // backtracked it, so its role has been fulfilled.
              next_base_decision = base_decision + 1;
            end
          end
        end
      end

      CHECK_WAIT: begin
        clearing = 0;
        for (int cuboid = 0; cuboid < CUBOIDS; cuboid++) begin
          if (potential_areas[cuboid] != run_stack_len) begin
            clearing = 1;
          end
        end

        if (!clearing | !TARGET_IGNORED_STATES) begin
          next_state = CHECK;
        end
      end

      CHECK: begin
        // All the actual logic for this state happens down below, when checking for
        // `next_state == CHECK`.
      end

      SOLUTION: begin
        // `decisions` is always meant to contain at least one 1 at the start for
        // running the first instruction, and so `decisions_len` should only ever be 0
        // while we're receiving.
        assert (decisions_len != 0);

        out_valid = 1;
        inc_decision_index = prefix_bits_left == 0 & out_ready;
        if (prefix_bits_left == 0 & decision_index == decisions_len - 1 | !TARGET_IGNORED_STATES) begin
          // We've just written out the last decision (if it was a decision in the first
          // place), and so we can go back to running.
          out_last = 1;
          if (out_ready | !TARGET_IGNORED_STATES) begin
            next_state = BACKTRACK;
          end
        end
      end

      STALL: begin
        next_state = saved_state;
      end

      SPLIT: begin
        assert (decisions_len != 0);
        out_valid = prefix_bits_left > 0 | decision_index < base_decision;

        if (prefix_bits_left == 0) begin
          if (decision_index == base_decision - 1) begin
            // Aha, here's the decision we're splitting on (the old `prefix.base_decision`;
            // we incremented it before entering SPLIT state).
            //
            // Change it to 0 for the new finder, and then leave it to try all the
            // combinations of the rest of the decisions.
            out_data = 0;
            out_last = 1;

            // We don't stop yet though, because `prefix.base_decision` might not point to a
            // 1 and so we need to keep going until we find one.
          end

          // Don't move forwards until either our output's been accepted or we're finished
          // transmitting.
          if (out_ready | !out_valid | !TARGET_IGNORED_STATES) begin
            inc_decision_index = 1;
            if (decision_index >= base_decision & decisions_rd_data == 1 | !TARGET_IGNORED_STATES) begin
              // Great, we've found a 1 to set `base_decision` to. Stop here.
              next_base_decision = decision_index;
              next_state = was_backtrack ? BACKTRACK : RUN;
            end else if (decision_index == decisions_len - 1) begin
              // We've gone through all the decisions without finding any 1s to make
              // `prefix.base_decision` point to, which means it needs to be 1 past the end of
              // `decisions`.
              next_base_decision = decisions_len;

              // It's a bit sketchy to increment `decision_index` past the end of `decisions`.
              inc_decision_index = 0;
              next_state = was_backtrack ? BACKTRACK : RUN;
            end
          end
        end
      end

      PAUSE: begin
        assert (decisions_len != 0);

        out_valid = 1;
        inc_decision_index = prefix_bits_left == 0 & out_ready;
        if (prefix_bits_left == 0 & decision_index == decisions_len - 1) begin
          // We've just written out the last decision (if it was a decision in the first
          // place), and so now it's time to clear everything out ready for the next
          // finder.
          out_last = 1;
          if (out_ready) begin
            local_reset = 1;
            // This is overriden by `local_reset` but eh, may as well specify it anyway.
            next_state  = CLEAR;
          end
        end
      end

      default: begin
        next_state = 'x;
        in_ready = 'x;
        run = 'x;
        local_reset = 'x;
        next_base_decision = 'x;
        out_valid = 'x;
        inc_decision_index = 'x;
        out_last = 'x;
        out_data = 'x;
      end
    endcase

    if (advance) begin
      if (run_stack_len == 0) begin
        // We're running the first instruction, and so the next one we should look at is
        // its first child.
        next_target_ref = '{parent: 0, child_index: 0};
      end else if (last_child) begin
        next_target_ref = '{parent: target_ref.parent + 1, child_index: 0};
      end else begin
        next_target_ref = '{parent: target_ref.parent, child_index: target_ref.child_index + 1};
      end
    end else if (backtrack) begin
      // We want to backtrack the last instruction in the run stack (which should
      // always be `target_parent` when backtracking), and then our next target is
      // the instruction after it in the queue.
      to_backtrack = target_parent.instruction_ref;
      if (to_backtrack.child_index == 3) begin
        // The instruction we're backtracking was the last child of its parent, and so
        // the instruction after it that we want to try next is the first child of the
        // next instruction.
        next_target_ref = '{parent: to_backtrack.parent + 1, child_index: 0};
      end else begin
        // Otherwise it's the next child of the same parent.
        next_target_ref = '{
            parent: to_backtrack.parent,
            child_index: to_backtrack.child_index + 1
        };
      end
    end else begin
      next_target_ref = target_ref;
    end

    if (state.solution | state.split | state.pause) begin
      // We're just reading from `decisions`.
      decisions_wr_data = 'x;
      decisions_wr_en = 0;
      next_decisions_len = decisions_len;
    end else if (state.backtrack) begin
      // To backtrack, we find the last 1 in `decisions`, turn it into a 0, and get
      // rid of all the decisions after it.
      // The last 1 is just the decision associated with the last instruction we ran
      // (which we always set `target_parent` when backtracking), and so we can
      // retrieve its index from there.
      last_run = target_parent.decision_index;
      decisions_wr_data = 0;
      // It's possible to be in BACKTRACK state without actually backtracking, if
      // `req_split` or `req_pause` is 1, and if that's the case we shouldn't write
      // anything.
      // We take this branch anyway so that `decisions_rd_addr` can be computed more
      // quickly.
      decisions_wr_en = backtrack;
      next_decisions_len = backtrack ? last_run + 1 : decisions_len;
    end else if (prefix_bits_left == 0 & in_valid & in_ready) begin
      decisions_wr_data = in_data;
      decisions_wr_en = 1;
      next_decisions_len = decisions_len + 1;
    end else if (advance & instruction_valid & |neighbours_valid) begin
      // This instruction is valid to run and isn't a potential instruction, which
      // means that whether or not we run it is a decision. Add it to the list.
      // Note that the only scenario where we don't run it here is when receiving a
      // finder.
      decisions_wr_data = run;
      decisions_wr_en = 1;
      next_decisions_len = decisions_len + 1;
    end else begin
      decisions_wr_data = 'x;
      decisions_wr_en = 0;
      next_decisions_len = decisions_len;
    end

    if (run) begin
      next_run_stack_len = run_stack_len_inc;
    end else if (backtrack) begin
      next_run_stack_len = run_stack_len_dec;
    end else begin
      next_run_stack_len = run_stack_len;
    end

    if (advance & instruction_valid & !(|neighbours_valid)) begin
      // We're advancing, and the instruction is valid but none of its neighbours are;
      // that means this is a potential instruction. Add it to the list.
      next_potential_len = potential_len + 1;
    end else if (backtrack) begin
      // Reset `potential_len` to whatever it was when the instruction we're
      // backtracking was run.
      next_potential_len = target_parent.potential_len;
    end else begin
      next_potential_len = potential_len;
    end

    for (int cuboid = 0; cuboid < CUBOIDS; cuboid++) begin
      automatic logic rd_data = potential_surface_rd_datas[cuboid];
      automatic logic wr_data;
      automatic logic wr_en;
      if (state.clear) begin
        wr_data = 0;
        wr_en   = int'(clear_index) < AREA;
      end else if (state.check) begin
        wr_data = 1;
        wr_en   = instruction_valid;
      end else begin
        wr_data = 0;
        wr_en   = int'(potential_surfaces_clear_index) < AREA;
      end

      potential_surface_wr_datas[cuboid] = wr_data;
      potential_surface_wr_ens[cuboid]   = wr_en;

      if (state.clear) begin
        // Don't update the count when in `CLEAR` state because it's currently bogus.
        // Just keep it as 0, which will eventually become correct once we're done
        // clearing.
        next_potential_areas[cuboid] = 0;
      end else begin
        next_potential_areas[cuboid] = potential_areas[cuboid];

        if (rd_data == 0 & wr_data == 1 & wr_en) begin
          next_potential_areas[cuboid] += 1;
          inc_potential_areas[cuboid] = 1;
        end else if (rd_data == 1 & wr_data == 0 & wr_en) begin
          next_potential_areas[cuboid] -= 1;
        end

        if (run) begin
          next_potential_areas[cuboid] += 1;
          // Note that we only set bits of `potential_surfaces` during CHECK state, and we
          // only run instructions during RECEIVE or RUN state, so we shouldn't have to
          // worry about adding 2 here.
          assert (!inc_potential_areas[cuboid]);
          inc_potential_areas[cuboid] = 1;
        end else if (backtrack) begin
          next_potential_areas[cuboid] -= 1;
        end
      end
    end

    if (next_state.run & next_target_ref.parent >= next_run_stack_len) begin
      // Hang on a minute, we can't go into RUN state - we'd be trying to run an
      // instruction past the end of the queue.
      //
      // Stop `target_ref` from becoming invalid.
      next_target_ref = target_ref;
      if (int'(run_stack_len) + int'(next_potential_len) >= AREA) begin
        // This has passed the first test for being a solution (there's enough run +
        // potential instructions to possibly cover the surfaces), so now we start the
        // next test of whether every square has at least 1 instruction that fills it.

        // Except if we're still clearing `potential_surfaces`. In that case we go into
        // CHECK_WAIT state until we're done.
        clearing = 0;
        for (int cuboid = 0; cuboid < CUBOIDS; cuboid++) begin
          if (potential_areas[cuboid] != run_stack_len) begin
            clearing = 1;
          end
        end

        if (clearing & TARGET_IGNORED_STATES) begin
          next_state = CHECK_WAIT;
        end else begin
          next_state = CHECK;
        end
      end else begin
        next_state = BACKTRACK;
      end
    end

    // This goes here instead of when we're already in `CHECK` state so that we
    // don't go into `CHECK` state when `potential_len == 0`.
    if (next_state.check) begin
      if (next_potential_index == next_potential_len) begin
        // We're about to advance to an invalid index, so stop now.
        maybe_solution = 1;
        for (int cuboid = 0; cuboid < CUBOIDS; cuboid++) begin
          // This is faster than using `next_potential_areas` because `AREA` is a
          // constant, so subtracting 1 from it costs nothing.
          //
          // We don't have to worry about `potential_areas[cuboid]` decreasing, that only
          // happens when clearing it which:
          // - Can't happen during CHECK state.
          // - Even if we're transitioning into CHECK state from RUN or CHECK_WAIT state,
          //   we only do so once `potential_areas[*] == run_stack_len`, so there's no
          //   clearing left that could be done in this clock cycle.
          if (
            int'(potential_areas[cuboid]) != AREA
            & !(int'(potential_areas[cuboid]) == AREA - 1 & inc_potential_areas[cuboid])
          ) begin
            maybe_solution = 0;
          end
        end

        if (maybe_solution & TARGET_IGNORED_STATES) begin
          reset_prefix_bits = 1;
          next_state = SOLUTION;
        end else begin
          next_state = BACKTRACK;
        end
      end else begin
        if (next_potential_index == potential_len) begin
          // The potential instruction at `next_potential_index` is only being added to
          // `potential_buf` on the next clock edge, so we can't get it from there;
          // instead just set it to `target_ref` directly.
          next_target_ref = target_ref;
        end else begin
          next_target_ref = next_potential_target;
        end
      end
    end

    if (next_state.backtrack) begin
      if (next_run_stack_len == 0) begin
        // We're trying to backtrack when there are no instructions left to backtrack;
        // that means we're done.
        local_reset = 1;
      end else begin
        // Make sure `target_parent` will be set to the last entry in the run stack.
        next_target_ref = '{parent: next_run_stack_len - 1, child_index: 'x};
      end
    end
  end
endmodule
