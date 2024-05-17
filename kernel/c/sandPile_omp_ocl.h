#include "sandPile.h"



unsigned ssandPile_compute_omp_ocl(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    int change = 0;

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        change |=
            do_tile(x + (x == 0), y + (y == 0),
                    TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                    TILE_H - ((y + TILE_H == DIM) + (y == 0)));
    swap_tables();
    if (change == 0)
      return it;
  }

  return 0;
}


#ifdef ENABLE_OPENCL

unsigned ssandPile_invoke_omp_ocl(unsigned nb_iter)
{
    size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y}; // global domain size for our calculation
    size_t local[2] = {TILE_W, TILE_H};          // local domain size for our calculation
    cl_int err;

    for (unsigned it = 1; it <= nb_iter; it++)
    {
        // Set kernel arguments
        //
        err = 0;
        err |= clSetKernelArg(compute_kernel, 0, sizeof(cl_mem), &next_buffer);
        err |= clSetKernelArg(compute_kernel, 1, sizeof(cl_mem), &cur_buffer);
        check(err, "Failed to set kernel arguments");

        err = clEnqueueNDRangeKernel(queue, compute_kernel, 2, NULL, global, local,
                                     0, NULL, NULL);
        check(err, "Failed to execute kernel");

        // Swap buffers
        {
            cl_mem tmp = cur_buffer;
            cur_buffer = next_buffer;
            next_buffer = tmp;
        }
    }

    clFinish(queue);

    return 0;
}

#endif
