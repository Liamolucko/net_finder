#include <stdbool.h>
#include <stdint.h>

#include "Vcore.h"
#include "verilated_vcd_c.h"

struct interface_inner {
    uint8_t *in_payload;
    bool *in_valid;
    const bool *in_ready;

    const uint8_t *out_payload;
    const bool *out_valid;
    bool *out_ready;

    bool *req_pause;
    bool *req_split;

    const bool *wants_finder;
    const bool *stepping;
};

struct neighbour_lookup_inner {
    bool *en;
    uint16_t *addr;
    uint64_t *data;
};

struct undo_lookup_inner {
    bool *en;
    uint16_t *addr;
    uint16_t *data;
};

struct core_inner {
    void *ptr;

    bool *clk;
    bool *reset;

    interface_inner interfaces[4];
    neighbour_lookup_inner neighbour_lookups[3];
    undo_lookup_inner undo_lookups[2];
};

extern "C" {
void *verilated_context_new(bool trace) {
    VerilatedContext *context = new VerilatedContext();
    context->traceEverOn(trace);
    return (void *)context;
}

uint64_t verilated_context_time(void *ptr) {
    VerilatedContext *context = (VerilatedContext *)ptr;
    return context->time();
}

void verilated_context_time_inc(void *ptr, uint64_t add) {
    VerilatedContext *context = (VerilatedContext *)ptr;
    context->timeInc(add);
}

void verilated_context_free(void *ptr) {
    VerilatedContext *context = (VerilatedContext *)ptr;
    delete context;
}

void *verilated_vcd_new() {
    return (void *)new VerilatedVcdC();
}

void verilated_vcd_open(void *ptr, const char *filename) {
    VerilatedVcdC *trace = (VerilatedVcdC *)ptr;
    trace->open(filename);
}

void verilated_vcd_dump(void *ptr, uint64_t timeui) {
    VerilatedVcdC *trace = (VerilatedVcdC *)ptr;
    trace->dump(timeui);
}

void verilated_vcd_flush(void *ptr) {
    VerilatedVcdC *trace = (VerilatedVcdC *)ptr;
    trace->flush();
}

void verilated_vcd_free(void *ptr) {
    VerilatedVcdC *trace = (VerilatedVcdC *)ptr;
    trace->close();
    delete trace;
}

struct core_inner core_new(void *context_ptr) {
    VerilatedContext *context = (VerilatedContext *)context_ptr;
    Vcore *core = new Vcore(context);
    return {
        .ptr = (void *)core,

        .clk = (bool *)&core->clk,
        .reset = (bool *)&core->rst,

        .interfaces =
            {
                {
                    .in_payload = &core->interfaces_0_input_payload,
                    .in_valid = (bool *)&core->interfaces_0_input_valid,
                    .in_ready = (bool *)&core->interfaces_0_input_ready,
                    .out_payload = &core->interfaces_0_output_payload,
                    .out_valid = (bool *)&core->interfaces_0_output_valid,
                    .out_ready = (bool *)&core->interfaces_0_output_ready,
                    .req_pause = (bool *)&core->interfaces_0_req_pause,
                    .req_split = (bool *)&core->interfaces_0_req_split,
                    .wants_finder = (bool *)&core->interfaces_0_wants_finder,
                    .stepping = (bool *)&core->interfaces_0_stepping,
                },
                {
                    .in_payload = &core->interfaces_1_input_payload,
                    .in_valid = (bool *)&core->interfaces_1_input_valid,
                    .in_ready = (bool *)&core->interfaces_1_input_ready,
                    .out_payload = &core->interfaces_1_output_payload,
                    .out_valid = (bool *)&core->interfaces_1_output_valid,
                    .out_ready = (bool *)&core->interfaces_1_output_ready,
                    .req_pause = (bool *)&core->interfaces_1_req_pause,
                    .req_split = (bool *)&core->interfaces_1_req_split,
                    .wants_finder = (bool *)&core->interfaces_1_wants_finder,
                    .stepping = (bool *)&core->interfaces_1_stepping,
                },
                {
                    .in_payload = &core->interfaces_2_input_payload,
                    .in_valid = (bool *)&core->interfaces_2_input_valid,
                    .in_ready = (bool *)&core->interfaces_2_input_ready,
                    .out_payload = &core->interfaces_2_output_payload,
                    .out_valid = (bool *)&core->interfaces_2_output_valid,
                    .out_ready = (bool *)&core->interfaces_2_output_ready,
                    .req_pause = (bool *)&core->interfaces_2_req_pause,
                    .req_split = (bool *)&core->interfaces_2_req_split,
                    .wants_finder = (bool *)&core->interfaces_2_wants_finder,
                    .stepping = (bool *)&core->interfaces_2_stepping,
                },
                {
                    .in_payload = &core->interfaces_3_input_payload,
                    .in_valid = (bool *)&core->interfaces_3_input_valid,
                    .in_ready = (bool *)&core->interfaces_3_input_ready,
                    .out_payload = &core->interfaces_3_output_payload,
                    .out_valid = (bool *)&core->interfaces_3_output_valid,
                    .out_ready = (bool *)&core->interfaces_3_output_ready,
                    .req_pause = (bool *)&core->interfaces_3_req_pause,
                    .req_split = (bool *)&core->interfaces_3_req_split,
                    .wants_finder = (bool *)&core->interfaces_3_wants_finder,
                    .stepping = (bool *)&core->interfaces_3_stepping,
                },
            },

        .neighbour_lookups =
            {
                {
                    .en = (bool *)&core->neighbour_lookups_0_en,
                    .addr = &core->neighbour_lookups_0_addr,
                    .data = &core->neighbour_lookups_0_data,
                },
                {
                    .en = (bool *)&core->neighbour_lookups_1_en,
                    .addr = &core->neighbour_lookups_1_addr,
                    .data = &core->neighbour_lookups_1_data,
                },
                {
                    .en = (bool *)&core->neighbour_lookups_2_en,
                    .addr = &core->neighbour_lookups_2_addr,
                    .data = &core->neighbour_lookups_2_data,
                },
            },

        .undo_lookups =
            {
                {
                    .en = (bool *)&core->undo_lookups_0_en,
                    .addr = &core->undo_lookups_0_addr,
                    .data = &core->undo_lookups_0_data,
                },
                {
                    .en = (bool *)&core->undo_lookups_1_en,
                    .addr = &core->undo_lookups_1_addr,
                    .data = &core->undo_lookups_1_data,
                },
            },
    };
}

void core_update(void *ptr) {
    Vcore *core = (Vcore *)ptr;
    core->eval();
}

void core_trace(void *ptr, void *trace_ptr) {
    Vcore *core = (Vcore *)ptr;
    // `levels` and `options` don't seem to do anything, so don't bother
    // exposing them.
    core->trace((VerilatedVcdC *)trace_ptr, 0);
}

void core_free(void *ptr) {
    Vcore *core = (Vcore *)ptr;
    core->final();
    delete core;
}
}

// According to Verilator's examples, 'Legacy function required only so linking
// works on Cygwin and MSVC++'. Ugh.
double sc_time_stamp() {
    return 0;
}
