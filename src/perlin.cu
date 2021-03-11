#include "hash.hh"
#include "util.hh"
#include "perlin.hh"

__device__
float interpolate(float a, float b, float w) {
  return (b - a) * w + a;
}

__device__
void perlin2d(BGR *out, int out_h, int out_w, int start_x, int stop_x, int start_y, int stop_y, int cell_size, uint64_t seed) {
  v2<int> gUL, gUR, gDL, gDR;
  v2<int> cell(cell_size, cell_size);

  float valscale = 127.0f / (cell_size * cell_size);
  int start_tx = start_x / cell_size;
  int start_ty = start_y / cell_size;
  int stop_tx = (stop_x + cell_size - 1) / cell_size;
  int stop_ty = (stop_y + cell_size - 1) / cell_size;
  for (int tx = start_tx; tx < stop_tx; tx++) {
    gUL = cell * hash2unit(coord_hash(seed, tx, start_ty));
    gUR = cell * hash2unit(coord_hash(seed, tx + 1, start_ty));

    int tile_x1 = tx == start_tx ? start_x : tx * cell_size;
    int tile_x2 = (tx + 1) == stop_tx ? stop_x : (tx + 1) * cell_size;

    for (int ty = start_ty; ty < stop_ty; ty++) {
      gDL = cell * hash2unit(coord_hash(seed, tx, ty + 1));
      gDR = cell * hash2unit(coord_hash(seed, tx + 1, ty + 1));

      int tile_y1 = ty == start_ty ? start_y : ty * cell_size;
      int tile_y2 = (ty + 1) == stop_ty ? stop_y : (ty + 1) * cell_size;

      for (int x = tile_x1; x < tile_x2; x++) {
        for (int y = tile_y1; y < tile_y2; y++) {
          v2<int> dUL(x - tx * cell_size, y - ty * cell_size),
              dUR(x - (tx + 1) * cell_size, y - ty * cell_size),
              dDL(x - tx * cell_size, y - (ty + 1) * cell_size),
              dDR(x - (tx + 1) * cell_size, y - (ty + 1) * cell_size);

          uint8_t v = (uint8_t)(
              valscale * interpolate(interpolate(gUL.dot(dUL), gUR.dot(dUR),
                                                 (float)dUL.x / cell_size),
                                     interpolate(gDL.dot(dDL), gDR.dot(dDR),
                                                 (float)dUL.x / cell_size),
                                     (float)dUL.y / cell_size) +
              127);
          BGR *elt = out + (y * out_w + x);
          elt->blue = elt->green = elt->red = v;
        }
      }

      gUL = gDL;
      gUR = gDR;
    }
  }
}
