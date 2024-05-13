#include "sandPile.h"
#include <fcntl.h>


#ifdef ENABLE_OPENCL

// Only called when --dump or --thumbnails is used
void ssandPile_refresh_img_ocl()
{
  printf("refresh\n");

  cl_int err;

  err =
      clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0,
                          sizeof(unsigned) * DIM * DIM, TABLE, 0, NULL, NULL);
  check(err, "Failed to read buffer from GPU");

  ssandPile_refresh_img();
}

void ssandPile_refresh_img_ocl_twin()
{
  ssandPile_refresh_img_ocl();
}

static cl_mem twin_buffer = 0;

// ./run -k ssandPile -g -v ocl_twin -a data/misc/mask.bin -i 69

void ssandPile_init_ocl_twin (void)
{
  ssandPile_init();
  const int size = DIM * DIM * sizeof (unsigned);

  // mask_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, NULL);
  // if (!mask_buffer)
  //   exit_with_error ("Failed to allocate mask buffer");
  
  printf("init\n");

  twin_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, NULL);
  if (!twin_buffer)
    exit_with_error ("Failed to allocate second input buffer");
}

void do_cpu_pre(){}

void do_cpu_post()
{
    cl_int err;
    err = clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0, sizeof(unsigned) * DIM * DIM, TABLE, 0, NULL, NULL);
    check(err, "Failed to read buffer from GPU");

    for(int i = 0; i < DIM; i ++)
    {
      for(int j = 0; j < 80; j++)
      {
        table(in, i, j) = (unsigned int)rand() % 4;

      }
    }

    err = clEnqueueWriteBuffer (queue,  cur_buffer, CL_TRUE, 0, sizeof(unsigned) * DIM * DIM, TABLE, 0, NULL, NULL);
    check(err, "Failed to write buffer to GPU");
    
    //swap_tables();
}


unsigned ssandPile_invoke_ocl_twin (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y}; // global domain size for our calculation
  size_t local[2]  = {TILE_W, TILE_H}; // local domain size for our calculation
  cl_int err;

  //do_cpu_pre();

  uint64_t clock = monitoring_start_tile(easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {

    //printf("it=%d\n", it);
    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &next_buffer);
    err |= clSetKernelArg (compute_kernel, 2, sizeof (cl_mem), &twin_buffer);
    //err |= clSetKernelArg (compute_kernel, 3, sizeof (cl_mem), &mask_buffer);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");

    // Swap buffers
    {
      // cl_mem tmp  = twin_buffer;
      // twin_buffer = next_buffer;
      // next_buffer = tmp;
      cl_mem tmp  = cur_buffer;
      cur_buffer  = next_buffer;
      next_buffer = tmp;
    }
  }

  clFinish (queue);

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  do_cpu_post();

  return 0;
}

#endif