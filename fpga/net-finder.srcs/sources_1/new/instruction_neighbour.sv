`include "types.svh"

`timescale 1ns / 1ps

function automatic instruction_t instruction_neighbour(instruction_t instruction,
                                                       logic [1:0] direction);
  instruction_t result;
  result.pos = instruction.pos;
  case (direction)
    0: result.pos.x = instruction.pos.x == 0 ? COORD_MAX : instruction.pos.x - 1;
    1: result.pos.y = instruction.pos.y == COORD_MAX ? 0 : instruction.pos.y + 1;
    2: result.pos.x = instruction.pos.x == COORD_MAX ? 0 : instruction.pos.x + 1;
    3: result.pos.y = instruction.pos.y == 0 ? COORD_MAX : instruction.pos.y - 1;
    default: ;
  endcase

  for (int cuboid = 0; cuboid < CUBOIDS; cuboid++) begin
    result.mapping[cuboid] = cursor_neighbour(cuboid, instruction.mapping[cuboid], direction);
  end

  return result;
endfunction
