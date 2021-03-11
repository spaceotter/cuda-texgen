#pragma once

#include <util.hh>

/**
 *  evaluate the noise function over the patch
 */
__device__
void perlin2d(BGR *out, int out_h, int out_w, int start_x, int stop_x, int start_y, int stop_y, int cell_size, uint64_t seed);
