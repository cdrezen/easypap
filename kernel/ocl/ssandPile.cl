#include "kernel/ocl/common.cl"


__kernel void ssandPile_ocl2 (__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  //TODO

  int pos = y * DIM + x;

  if (x != 0 && y != 0 && x != DIM-1 && y != DIM-1)  //pour le pas calculer les tuiles des bords
  {
    // unsigned cell_out = in[pos] % 4;
    // cell_out += in[pos + 1] / 4;
    // cell_out += in[pos - 1] / 4;
    // cell_out += in[pos + DIM] / 4;
    // cell_out += in[pos - DIM] / 4;
    // out[pos] = cell_out;

    unsigned cell_out = in[pos] & 3
                      + in[pos + 1] >> 2 
                      + in[pos - 1] >> 2
                      + in[pos + DIM] >> 2
                      + in[pos - DIM] >> 2;
    out[pos] = cell_out;
  }
}

__kernel void ssandPile_ocl(__global unsigned *in, __global unsigned *out) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int pos = y * DIM + x;

    if (x != 0 && y != 0 && x != DIM - 1 && y != DIM - 1) {
        unsigned cell_out = in[pos] % 4
                          + in[pos + 1] / 4 
                          + in[pos - 1] / 4
                          + in[pos + DIM] / 4
                          + in[pos - DIM] / 4;
        out[pos] = cell_out;
    }
}


// DO NOT MODIFY: this kernel updates the OpenGL texture buffer
// This is a ssandPile-specific version (generic version is defined in common.cl)
__kernel void ssandPile_update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  int2 pos = (int2)(x, y);
  unsigned c = cur [y * DIM + x];
  unsigned r = 0, v = 0, b = 0;

  if (c == 1)
    v = 255;
  else if (c == 2)
    b = 255;
  else if (c == 3)
    r = 255;
  else if (c == 4)
    r = v = b = 255;
  else if (c > 4)
    r = v = b = (2 * c);

  c = rgba(r, v, b, 0xFF);
  write_imagef (tex, pos, color_scatter (c));
}
