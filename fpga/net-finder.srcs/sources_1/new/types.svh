`ifndef TYPES_SVH
`define TYPES_SVH

`include "generated.svh"

// Round the size of the net up to the nearest multiple of 4 so that our tiling works properly.
parameter int NET_SIZE = 4 * int'($ceil($itor(AREA) / 4));
parameter int COORD_BITS = $clog2(NET_SIZE);
parameter int NET_LEN = NET_SIZE << COORD_BITS;

typedef logic [COORD_BITS-1:0] coord_t;
parameter coord_t COORD_MAX = coord_t'(NET_SIZE - 1);

typedef struct packed {
  coord_t x;
  coord_t y;
} pos_t;
typedef struct packed {
  pos_t pos;
  mapping_t mapping;
} instruction_t;

`endif
