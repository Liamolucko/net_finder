#include <stdbool.h>

#include "Vcore.h"

struct core_inner {
    void *ptr;

    bool *clk;
    bool *reset;

    bool *in_data;
    bool *in_valid;
    const bool *in_ready;
    bool *in_last;

    const bool *out_data;
    const bool *out_valid;
    bool *out_ready;
    const bool *out_last;

    const bool *out_solution;
    const bool *out_split;
    const bool *out_pause;

    bool *req_pause;
    bool *req_split;

    const bool *stepping;
};

extern "C" {
void *verilated_context_new(bool trace) {
    VerilatedContext *context = new VerilatedContext();
    context->traceEverOn(trace);
    return (void *)context;
}

void verilated_context_time_inc(void *ptr, uint64_t add) {
    VerilatedContext *context = (VerilatedContext *)ptr;
    context->timeInc(add);
}

void verilated_context_free(void *ptr) {
    VerilatedContext *context = (VerilatedContext *)ptr;
    delete context;
}

struct core_inner core_new(void *context_ptr) {
    VerilatedContext *context = (VerilatedContext *)context_ptr;
    Vcore *core = new Vcore(context);
    return {
        .ptr = (void *)core,

        .clk = (bool *)&core->clk,
        .reset = (bool *)&core->reset,
        .in_data = (bool *)&core->in_data,
        .in_valid = (bool *)&core->in_valid,
        .in_ready = (bool *)&core->in_ready,
        .in_last = (bool *)&core->in_last,
        .out_data = (bool *)&core->out_data,
        .out_valid = (bool *)&core->out_valid,
        .out_ready = (bool *)&core->out_ready,
        .out_last = (bool *)&core->out_last,
        .out_solution = (bool *)&core->out_solution,
        .out_split = (bool *)&core->out_split,
        .out_pause = (bool *)&core->out_pause,
        .req_pause = (bool *)&core->req_pause,
        .req_split = (bool *)&core->req_split,
        .stepping = (bool *)&core->stepping,
    };
}

void core_update(void *ptr) {
    Vcore *core = (Vcore *)ptr;
    core->eval();
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
