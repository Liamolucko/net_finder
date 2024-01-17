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

// A simple-dual-port, write-before-read, synchronous RAM for storing run stacks.
module run_stack_ram (
    input logic clk,

    input run_stack_index_t wr_addr,
    input run_stack_entry_t wr_data,
    input logic wr_en,

    input  run_stack_index_t rd_addr,
    output run_stack_entry_t rd_data
);
  logic [$bits(run_stack_entry_t)-1:0] memory[AREA];

  run_stack_entry_t unsynced_rd_data;
  run_stack_entry_t last_wr_data;
  logic conflict;
  always_ff @(posedge clk) begin
    unsynced_rd_data <= memory[rd_addr];
    if (wr_en) begin
      memory[wr_addr] <= wr_data;
    end

    last_wr_data <= wr_data;
    conflict <= wr_en & wr_addr == rd_addr;
  end

  // Note: I don't think this can use 7 series BRAMs' `WRITE_FIRST` mode because
  // that only applies when you're reading and writing on the same port. In this
  // case reads & writes use different ports and so we need to do it manually.
  assign rd_data = conflict ? last_wr_data : unsynced_rd_data;
endmodule

module core (
    input logic clk,
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

    // Whether the next clock cycle will be put to use for either running or
    // backtracking. For testing purposes.
    output logic stepping
);
  typedef enum {
    // The core is currently clearing out all its internal memory after a reset.
    CLEAR,
    // The core is currently receiving or waiting to receive a finder to run, and
    // filling in its state.
    RECEIVE,
    // The core is actively attempting to run instructions.
    RUN,
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
    BACKTRACK,
    // The core is waiting for its leftover state from the last `CHECK` to be
    // cleared out before switching into `CHECK` state.
    CHECK_WAIT,
    // The core is checking if its current solution is valid.
    CHECK,
    // The core is outputting a potential solution.
    SOLUTION,
    // The core has split its finder in half and is currently outputting one of
    // those halves.
    SPLIT,
    // The core is being paused; it's currently writing out the state of its finder.
    // Once it's done, it'll switch to `receive` state.
    PAUSE
  } state_t;
  state_t state;
  state_t next_state;

  always_ff @(posedge clk, posedge reset) begin
    if (reset | sync_reset) begin
      state <= CLEAR;
    end else begin
      state <= next_state;
    end
  end

  assign out_solution = out_valid & state == SOLUTION;
  assign out_split = out_valid & state == SPLIT;
  assign out_pause = out_valid & state == PAUSE;

  // Control signals.
  // Whether to do a synchronous reset on the next clock edge.
  logic sync_reset;
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

  // Collect that metadata into a shift register.
  prefix_t prefix;
  logic [$clog2($bits(prefix)+1)-1:0] prefix_bits_left;

  // What we're going to set `prefix.base_decision` to next clock cycle (if
  // `prefix` isn't currently halfway through being shifted in/out of the core).
  decision_index_t next_base_decision;

  always_ff @(posedge clk or posedge reset) begin
    if (reset | sync_reset | reset_prefix_bits) begin
      prefix_bits_left <= $bits(prefix_t);
    end else if (prefix_bits_left > 0 & (
      (!sending & in_valid & in_ready)
      | (sending & out_valid & out_ready)
    )) begin
      prefix_bits_left <= prefix_bits_left - 1;
    end
  end

  always_ff @(posedge clk) begin
    // We don't bother clearing `prefix` on reset since it's bogus anyway.
    if (prefix_bits_left > 0 & (
      (!sending & in_valid & in_ready)
      | (sending & out_valid & out_ready)
    )) begin
      prefix <= {prefix[$bits(prefix)-2:0], sending ? prefix[$bits(prefix)-1] : in_data};
    end else if (prefix_bits_left == 0) begin
      prefix.base_decision <= next_base_decision;
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
      .rd_data()
  );

  // The number of decisions we've made.
  logic [$clog2(4*AREA+1)-1:0] decisions_len;
  logic [$clog2(4*AREA+1)-1:0] next_decisions_len;

  // stupid broken-out signals for the ports of `decisions`
  decision_index_t decisions_addr;
  logic decisions_wr_en;
  logic decisions_wr_data;

  always_ff @(posedge clk or posedge reset) begin
    if (reset | sync_reset) begin
      decisions_len <= 0;
    end else begin
      decisions_len <= next_decisions_len;
    end
  end

  always_comb begin
    automatic decision_index_t last_run = 'x;

    if (prefix_bits_left == 0 & in_valid & in_ready) begin
      decisions_addr = decisions_len;
      decisions_wr_data = in_data;
      decisions_wr_en = 1;
      next_decisions_len = decisions_len + 1;
    end else if (advance & vc.instruction_valid & |vc.neighbours_valid) begin
      // This instruction is valid to run and isn't a potential instruction, which
      // means that whether or not we run it is a decision. Add it to the list.
      // Note that the only scenario where we don't run it here is when receiving a
      // finder.
      decisions_addr = decisions_len;
      decisions_wr_data = run;
      decisions_wr_en = 1;
      next_decisions_len = decisions_len + 1;
    end else if (backtrack) begin
      // To backtrack, we find the last 1 in `decisions`, turn it into a 0, and get
      // rid of all the decisions after it.
      // The last 1 is just the decision associated with the last instruction we ran
      // (which should always be `target_parent` since we only backtrack when at the
      // end of the queue), and so we can retrieve its index from there.
      last_run = target_parent.decision_index;
      decisions_addr = last_run;
      decisions_wr_data = 0;
      decisions_wr_en = 1;
      next_decisions_len = last_run + 1;
    end else begin
      decisions_addr = decision_index;
      decisions_wr_data = 'x;
      decisions_wr_en = 0;
      next_decisions_len = decisions_len;
    end
  end

  // The index of the current decision we're sending.
  decision_index_t decision_index;
  always_ff @(posedge clk or posedge reset) begin
    if (reset | sync_reset | (state != SOLUTION & state != SPLIT & state != PAUSE)) begin
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
      .rd_addr(next_target_ref.parent),
      .rd_data(target_parent)
  );
  logic [$clog2(AREA+1)-1:0] run_stack_len;
  logic [$clog2(AREA+1)-1:0] next_run_stack_len;

  // A reference to the instruction we're currently looking at.
  //
  // This always lines up with `target`, except when `run_stack_len` is 0 and in
  // `CHECK` state.
  instruction_ref_t target_ref;
  // The parent of `target`'s entry in the run stack.
  run_stack_entry_t target_parent;

  // The instruction we're currently looking at.
  instruction_t target;
  // The direction of `target` from `target_parent`.
  logic [1:0] target_direction;
  // Whether `target` is the last child of `target_parent` we have to look at.
  logic last_child;

  // The instruction we're going to switch to looking at on the next clock edge.
  instruction_ref_t next_target_ref;

  always_ff @(posedge clk or posedge reset) begin
    if (reset | sync_reset) begin
      run_stack_len <= 0;
      // Don't bother resetting `target_ref`, it's nonsense until `run_stack_len` is
      // at least 1 anyway.
    end else begin
      run_stack_len <= next_run_stack_len;
      target_ref <= next_target_ref;
    end
  end

  always_comb begin
    if (run) begin
      next_run_stack_len = run_stack_len + 1;
    end else if (backtrack) begin
      next_run_stack_len = run_stack_len - 1;
    end else begin
      next_run_stack_len = run_stack_len;
    end
  end

  always_comb begin
    automatic logic [1:0] last_valid_direction = 'x;
    automatic int current_child = 'x;

    if (run_stack_len == 0) begin
      // We must still be in the process of receiving a finder, and so our current
      // target has the mapping we've received and a default net position of (0, 0).
      target = '{pos: '{x: 0, y: 0}, mapping: prefix.start_mapping};
      // These are nonsense, since `target` is the first instruction and doesn't have
      // a parent.
      target_direction = 'x;
      last_child = 'x;
    end else begin
      // Find the direction of the last valid child.
      assert (run_stack_len == 0 | target_parent.children != 'b0000);
      last_valid_direction = 'x;
      for (int direction = 3; direction >= 0; direction--) begin
        if (target_parent.children[direction]) begin
          last_valid_direction = direction[1:0];
          break;
        end
      end

      // To start off with, set `target_direction` to the last valid direction. That
      // way, if `child_index` is past the end of the list of valid children, it
      // defaults to the last one.
      target_direction = last_valid_direction;

      // Then check if there's an exact match.
      current_child = -1;
      for (int direction = 0; direction < 4; direction++) begin
        if (target_parent.children[direction]) begin
          // The child is valid so it actually gets a new index.
          current_child += 1;
          if (current_child == int'(target_ref.child_index)) begin
            target_direction = direction[1:0];
            break;
          end
        end
      end

      target = instruction_neighbour(target_parent.instruction, target_direction);
      last_child = target_direction == last_valid_direction;
    end
  end

  // The list of potential instructions.
  // I would've just named it `potential` but that's a keyword in SystemVerilog.
  instruction_ref_t potential_buf[4 * AREA];

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
  // The length of `potential_buf`.
  //
  // In theory might need more bits than `potential_index_t`, except that it's
  // impossible for the length of `potential_buf` to max out like that anyway
  // since it would require the first instruction to be a potential instruction.
  // And doing it this way guarantees that this uses the same number of bits as a
  // cursor.
  potential_index_t potential_len;
  potential_index_t next_potential_len;

  always_ff @(posedge clk or posedge reset) begin
    if (reset | sync_reset) begin
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

  always_comb begin
    if (advance & vc.instruction_valid & !(|vc.neighbours_valid)) begin
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
  end

  always_comb begin
    if (state == CHECK) begin
      next_potential_index = potential_index + 1;
    end else begin
      next_potential_index = 0;
    end
  end

  // When in `CLEAR` state, the index in the net/surface we're currently clearing.
  logic [2*COORD_BITS-3:0] clear_index;
  always_ff @(posedge clk, posedge reset) begin
    if (reset | sync_reset | state != CLEAR) begin
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

      .instruction(backtrack ? target_parent.instruction : target),
      .start_mapping_index(prefix.start_mapping_index),

      .run(run),
      .backtrack(backtrack),

      .neighbours_valid (),
      .instruction_valid(),

      .neighbours_valid_in(target_parent.children),

      .instruction_cursors_filled(),

      .clear_mode (state == CLEAR),
      .clear_index(clear_index)
  );

  // verilator lint_off ASSIGNIN
  // verilator lint_off PINMISSING

  // Instantiate RAMs used for keeping track of which squares on the surfaces are
  // filled by potential instructions.
  //
  // Most of the time they're maintained as all 0s, and then they're filled in in
  // `CHECK` state.
  // First we declare their signals, then actually instantiate them inside a `generate`.

  // How many bits are set on each surface in `potential_surfaces`.
  logic [$clog2(AREA+1)-1:0] potential_surface_counts[CUBOIDS];
  logic [$clog2(AREA+1)-1:0] next_potential_surface_counts[CUBOIDS];
  always_ff @(posedge clk or posedge reset) begin
    if (reset | sync_reset) begin
      for (int cuboid = 0; cuboid < CUBOIDS; cuboid++) begin
        // We don't actually have any idea how many 1s there are at this point, but
        // since we go into `CLEAR` state on reset 0 will become correct in a second
        // anyway.
        potential_surface_counts[cuboid] <= 0;
      end
    end else begin
      potential_surface_counts <= next_potential_surface_counts;
    end
  end

  // Once we exit `CHECK` state, we need to reset `potential_surfaces` to all 0s
  // ready for the next time we enter `CHECK` state. This is the index we're up to
  // in doing so.
  square_t potential_surfaces_clear_index;
  always_ff @(posedge clk or posedge reset) begin
    if (reset | sync_reset | state == CHECK) begin
      potential_surfaces_clear_index <= 0;
    end else begin
      potential_surfaces_clear_index <= potential_surfaces_clear_index + 1;
    end
  end

  generate
    genvar cuboid;
    for (cuboid = 0; cuboid < CUBOIDS; cuboid++) begin : gen_potential_surfaces
      surface ram (
          .clk(clk),
          .rw_addr(
            state == CLEAR ? square_t'(clear_index)
            : state == CHECK ? target.mapping[cuboid].square
            : potential_surfaces_clear_index
          ),
          // While in `CHECK` state, we always want to write 1s for new squares filled by
          // potential instructions.
          // Otherwise, we're clearing and want to write 0s.
          .rw_wr_data(state == CHECK),
          // When in `CHECK` state, we only want to write a 1 if this square isn't already
          // filled by run instructions.
          // Otherwise we always want to write 0s (as long as the address is valid).
          .rw_wr_en(
            state == CLEAR ? int'(clear_index) < AREA
            : state == CHECK ? vc.instruction_valid & !vc.instruction_cursors_filled[cuboid]
            : int'(potential_surfaces_clear_index) < AREA
          )
      );

      always_comb begin
        if (state == CLEAR) begin
          // Don't update the count when in `CLEAR` state because it's currently bogus.
          // Just keep it as 0, which will eventually become correct once we're done
          // clearing.
          next_potential_surface_counts[cuboid] = 0;
        end else if (ram.rw_rd_data == 0 & ram.rw_wr_data == 1 & ram.rw_wr_en) begin
          next_potential_surface_counts[cuboid] = potential_surface_counts[cuboid] + 1;
        end else if (ram.rw_rd_data == 1 & ram.rw_wr_data == 0 & ram.rw_wr_en) begin
          next_potential_surface_counts[cuboid] = potential_surface_counts[cuboid] - 1;
        end else begin
          next_potential_surface_counts[cuboid] = potential_surface_counts[cuboid];
        end
      end
    end
  endgenerate
  // verilator lint_on ASSIGNIN
  // verilator lint_on PINMISSING

  // When in SPLIT state, whether we were in BACKTRACK state before switching to
  // SPLIT state.
  logic was_backtrack;
  always_ff @(posedge clk) begin
    if (state != SPLIT & next_state == SPLIT) begin
      was_backtrack <= state == BACKTRACK;
    end
  end

  assign out_valid = sending;

  // Use one big `always_comb` for most of our logic so that later steps can rely
  // on the results of earlier steps without re-triggering earlier steps and
  // creating cycles.
  always_comb begin
    in_ready = 0;
    out_last = 'x;
    // This is almost always what it should be set to, except when we override it to 0 during splitting.
    out_data = prefix_bits_left > 0 ? prefix[$bits(prefix)-1] : decisions.rd_data;

    sync_reset = 0;
    advance = 0;
    run = 0;
    backtrack = 0;
    inc_decision_index = 0;
    sending = 0;
    reset_prefix_bits = 0;

    // By default, stay in the current state.
    next_state = state;
    next_base_decision = prefix.base_decision;

    case (state)
      CLEAR: begin
        if (int'(clear_index) == NET_LEN / 4 - 1) begin
          // We're about to clear the last bit of the net, in which case we're done clearing.
          next_state = RECEIVE;
        end
      end

      RECEIVE: begin
        if (prefix_bits_left == 0) begin
          if (!vc.instruction_valid | !(|vc.neighbours_valid)) begin
            // No need to wait until the next decision is read, this is either invalid or a
            // potential instruction anyway so we can just advance past it.
            advance = 1;
          end else if (in_valid) begin
            // This instruction is valid and we've received a decision for it; do as it
            // says.
            advance = 1;
            in_ready = 1;
            run = in_data;
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

        if (req_pause) begin
          // We always pause immediately upon request.
          reset_prefix_bits = 1;
          next_state = PAUSE;
        end else if (req_split & prefix.base_decision < decisions_len) begin
          // We only split if `base_decision` is within `decisions`, since we need to
          // split on it. If it isn't, we continue until either:
          // - An instruction is run, at which point `base_decision` now points to the
          //   decision to do so and we can split on it.
          // - We get to the end of the queue without running anything; that means
          //   `base_decision` is still 1 past the end of `decisions`, and thus there are
          //   no decisions we're allowed to backtrack and we're done.
          reset_prefix_bits = 1;
          // Increment `base_decision` now, since we immediately start sending `prefix`
          // and need it to be correct.
          //
          // This isn't necessarily what we actually want the new `prefix.base_decision`
          // to be, but we correct that later.
          next_base_decision = prefix.base_decision + 1;
          next_state = SPLIT;
        end else begin
          // We always want to try and advance; but if it turns out we need to backtrack,
          // we change this back to 0 in a sec.
          advance = 1;
          if (vc.instruction_valid & |vc.neighbours_valid) begin
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

        if (req_pause) begin
          // We always pause immediately upon request.
          reset_prefix_bits = 1;
          next_state = PAUSE;
        end else if (req_split & prefix.base_decision < decisions_len) begin
          reset_prefix_bits = 1;
          next_base_decision = prefix.base_decision + 1;
          next_state = SPLIT;
        end else begin
          if (target_parent.decision_index < prefix.base_decision) begin
            // Hold up, we're about to try and backtrack an decision that it isn't this
            // finder's job to try both options of. That means we're done!
            sync_reset = 1;
          end else begin
            backtrack  = 1;
            next_state = RUN;

            if (target_parent.decision_index == prefix.base_decision) begin
              // The base decision is being backtracked, and so it doesn't really make sense
              // to call it the base decision anymore. What being the base decision means is
              // that it's the earliest decision that we need to backtrack; but we've just
              // backtracked it, so its role has been fulfilled.
              next_base_decision = prefix.base_decision + 1;
            end
          end
        end
      end

      CHECK_WAIT: begin
        automatic logic clearing = 0;
        for (int cuboid = 0; cuboid < CUBOIDS; cuboid++) begin
          if (next_potential_surface_counts[cuboid] != 0) begin
            clearing = 1;
          end
        end

        if (!clearing) begin
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

        sending = 1;
        inc_decision_index = prefix_bits_left == 0;
        if (prefix_bits_left == 0 & decision_index == decisions_len - 1) begin
          // We've just written out the last decision (if it was a decision in the first
          // place), and so we can go back to running.
          out_last   = 1;
          next_state = BACKTRACK;
        end
      end

      SPLIT: begin
        assert (decisions_len != 0);
        inc_decision_index = prefix_bits_left == 0;
        sending = prefix_bits_left > 0 | decision_index < prefix.base_decision;

        if (prefix_bits_left == 0) begin
          if (decision_index == prefix.base_decision - 1) begin
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

          if (decision_index >= prefix.base_decision & decisions.rd_data == 1) begin
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

      PAUSE: begin
        assert (decisions_len != 0);

        sending = 1;
        inc_decision_index = prefix_bits_left == 0;
        if (prefix_bits_left == 0 & decision_index == decisions_len - 1) begin
          // We've just written out the last decision (if it was a decision in the first
          // place), and so now it's time to clear everything out ready for the next
          // finder.
          out_last   = 1;
          sync_reset = 1;
          // This is overriden by `sync_reset` but eh, may as well specify it anyway.
          next_state = CLEAR;
        end
      end
      default: ;
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
      automatic instruction_ref_t to_backtrack = target_parent.instruction_ref;
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

    if (next_state == RUN & next_target_ref.parent >= next_run_stack_len) begin
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
        automatic logic clearing = 0;
        for (int cuboid = 0; cuboid < CUBOIDS; cuboid++) begin
          if (next_potential_surface_counts[cuboid] != 0) begin
            clearing = 1;
          end
        end

        if (clearing) begin
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
    if (next_state == CHECK) begin
      if (next_potential_index >= next_potential_len) begin
        // We're about to advance to an invalid index, so stop now.
        automatic logic maybe_solution = 1;
        for (int cuboid = 0; cuboid < CUBOIDS; cuboid++) begin
          // Use `next_potential_surface_counts` to make sure we take into account the
          // last potential instruction.
          if (int'(run_stack_len) + int'(next_potential_surface_counts[cuboid]) != AREA) begin
            maybe_solution = 0;
          end
        end

        if (maybe_solution) begin
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
          next_target_ref = potential_buf[next_potential_index];
        end
      end
    end

    if (next_state == BACKTRACK) begin
      if (next_run_stack_len == 0) begin
        // We're trying to backtrack when there are no instructions left to backtrack;
        // that means we're done.
        sync_reset = 1;
      end else begin
        // Make sure `target_parent` will be set to the last entry in the run stack.
        next_target_ref = '{parent: next_run_stack_len - 1, child_index: 'x};
      end
    end
  end
endmodule
