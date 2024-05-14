#include "sandPile.h"
#include <fcntl.h>


#ifdef ENABLE_OPENCL

#define in_buff cur_buffer
#define out_buff next_buffer

const double CPU_PROPORTION = 0.5;
//static cl_mem twin_buffer = 0;
// double cpu_tiles_x = 0;
// int border_left = 0;
// int border_right = 0;

double cpu_tiles_y = 0;
int border_top = 0;
int border_bottom = 0;


// Only called when --dump or --thumbnails is used
void ssandPile_refresh_img_ocl()
{
  printf("refresh\n");

  cl_int err;

  err =
      clEnqueueReadBuffer(queue, in_buff, CL_TRUE, 0,
                          sizeof(unsigned) * DIM * DIM, TABLE, 0, NULL, NULL);
  check(err, "Failed to read buffer from GPU");

  ssandPile_refresh_img();
}

void ssandPile_refresh_img_ocl_omp()
{
  ssandPile_refresh_img_ocl();
}

// ./run -k ssandPile -g -v ocl_omp -a data/misc/mask.bin -i 69

void ssandPile_init_ocl_omp (void)
{
  ssandPile_init();
  const int size = DIM * DIM * sizeof (unsigned);

  // cpu_tiles_x = CPU_PROPORTION * NB_TILES_X;
  // border_left = (cpu_tiles_x / 2) * TILE_W;
  // border_right = DIM - (cpu_tiles_x / 2) * TILE_W;
  
  cpu_tiles_y = CPU_PROPORTION * NB_TILES_Y;
  border_top = (cpu_tiles_y / 2) * TILE_H;
  border_bottom = DIM - (cpu_tiles_y / 2) * TILE_H;

  printf("init t=%d b=%d nb/2=%f\n", border_top, border_bottom, (cpu_tiles_y / 2));

  // mask_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, NULL);
  // if (!mask_buffer)
  //   exit_with_error ("Failed to allocate mask buffer");
  
  // twin_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, NULL);
  // if (!twin_buffer)
  //   exit_with_error ("Failed to allocate second input buffer");
}

//dbg
int ssandPile_do_tile_dbg(int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++) {
    for (int j = x; j < x + width; j++)
    {
        //table(in, i, j) = (unsigned int)rand() % 4;
        table(in, i, j) = 3;
    }
  }

  return 1;
}

void do_cpu_pre(){}

void do_cpu_post()
{
    cl_int err;
    err = clEnqueueReadBuffer(queue, in_buff, CL_TRUE, 0, sizeof(unsigned) * DIM * DIM, TABLE, 0, NULL, NULL);
    check(err, "Failed to read buffer from GPU");

    err = clEnqueueReadBuffer(queue, out_buff, CL_TRUE, 0, sizeof(unsigned) * DIM * DIM, table_cell(TABLE, out, 0, 0), 0, NULL, NULL);
    check(err, "Failed to read buffer from GPU");

    int change = 0;

    #pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change)
    for(int y = 0; y < DIM; y+=TILE_H) {
      for(int x = 0; x < DIM; x+=TILE_W)
      {
        //if(x > border_left && x < border_right) continue;
        if(y > border_top && y < border_bottom) continue;
        //if(is_steady(in, y, x) && is_steady(out, y, x)) continue;

        int diff = do_tile(x + (x == 0), y + (y == 0),
                     TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                     TILE_H - ((y + TILE_H == DIM) + (y == 0)));

        change |= diff;
      }
    }

    //swap_tables();

    err = clEnqueueWriteBuffer (queue,  out_buff, CL_TRUE, 0, sizeof(unsigned) * DIM * DIM, table_cell(TABLE, out, 0, 0), 0, NULL, NULL);
    check(err, "Failed to write buffer to GPU");

    // Swap buffers
    {
      cl_mem tmp  = in_buff;
      in_buff = out_buff;
      out_buff = tmp;
    }

    //swap_tables();
}


unsigned ssandPile_invoke_ocl_omp (unsigned nb_iter)
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
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &in_buff);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &out_buff);
    err |= clSetKernelArg (compute_kernel, 2, sizeof (unsigned), &border_top);
    err |= clSetKernelArg (compute_kernel, 3, sizeof (unsigned), &border_bottom);

    //err |= clSetKernelArg (compute_kernel, 3, sizeof (cl_mem), &mask_buffer);
    // check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                   0, NULL, NULL);

    check (err, "Failed to execute kernel");

    do_cpu_post();
  }

  clFinish (queue);

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  return 0;
}

#endif