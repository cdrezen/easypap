#include "sandPile.h"
#include <fcntl.h>


#ifdef ENABLE_OPENCL

#define in_buff cur_buffer
#define out_buff next_buffer

const double CPU_GPU_RATIO = 0.5;

double cpu_tiles_y = 0;
int border_top = 0;
int border_bottom = 0;
// double cpu_tiles_x = 0;
// int border_left = 0;
// int border_right = 0;


// Only called when --dump or --thumbnails is used
void ssandPile_refresh_img_ocl()
{
  printf("refresh\n");

  cl_int err;

  err = clEnqueueReadBuffer(queue, in_buff, CL_TRUE, border_top * DIM * sizeof(unsigned),
                          sizeof(unsigned) * border_top * DIM, &table(in, border_top, 0), 0, NULL, NULL);
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

  // cpu_tiles_x = CPU_GPU_RATIO * NB_TILES_X;
  // border_left = (cpu_tiles_x / 2) * TILE_W;
  // border_right = DIM - (cpu_tiles_x / 2) * TILE_W;
  
  // cpu_tiles_y = CPU_GPU_RATIO * NB_TILES_Y;
  // border_top = (cpu_tiles_y / 2) * TILE_H;
  // border_bottom = DIM - (cpu_tiles_y / 2) * TILE_H;

  cpu_tiles_y = CPU_GPU_RATIO * NB_TILES_Y;
  border_top = cpu_tiles_y * TILE_H;
  
  printf("init t=%d b=%d nb/2=%f\n", border_top, border_bottom, (cpu_tiles_y / 2));
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
    const unsigned int NB_LINES = 1;

    //read gpu line at border
    cl_int err = 0;
    const size_t offset_gpu = border_top * DIM * sizeof(unsigned);
    const size_t size = sizeof(unsigned) * (NB_LINES) * DIM;
    const TYPE* dest = &table(in, border_top, 0);
    err |= clEnqueueReadBuffer(queue, in_buff, CL_TRUE, offset_gpu, size, dest, 0, NULL, NULL);
    check(err, "Failed to read buffer from GPU");

    int change = 0;

      #pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change)
    for(int y = 0; y < border_top; y+=TILE_H) {
      for(int x = 0; x < DIM; x+=TILE_W)
      {
        //if(is_steady(in, y, x) && is_steady(out, y, x)) continue;

        int diff = do_tile(x + (x == 0), y + (y == 0),
                     TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                     TILE_H - ((y + TILE_H == DIM) + (y == 0)));

        change |= diff;
      }
    }

    //write cpu line at border
    const size_t offset_cpu = (border_top - NB_LINES) * DIM * sizeof(unsigned);
    const TYPE* source = &table(out, border_top - NB_LINES, 0);
    err |= clEnqueueWriteBuffer (queue,  out_buff, CL_TRUE, offset_cpu, size, source, 0, NULL, NULL);
    check(err, "Failed to write buffer to GPU");

    // Swap buffers
    {
      cl_mem tmp  = in_buff;
      in_buff = out_buff;
      out_buff = tmp;
      swap_tables();
    }
}


unsigned ssandPile_invoke_ocl_omp (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y}; // global domain size for our calculation
  size_t local[2]  = {TILE_W, TILE_H}; // local domain size for our calculation
  cl_int err;


  uint64_t clock = monitoring_start_tile(easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {

    //printf("it=%d\n", it);

    //do_cpu_pre();

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &in_buff);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &out_buff);
    err |= clSetKernelArg (compute_kernel, 2, sizeof (unsigned), &border_top);
    err |= clSetKernelArg (compute_kernel, 3, sizeof (unsigned), &border_bottom);

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                   0, NULL, NULL);

    check (err, "Failed to execute kernel");

    do_cpu_post();
  }

  clFinish (queue);

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  if(do_display)
  {
    err |= clEnqueueWriteBuffer(queue, in_buff, CL_TRUE, 0,
                          sizeof(unsigned) * border_top * DIM, &table(in, 0, 0), 0, NULL, NULL);
    check(err, "Failed to write buffer from GPU");
  }

  return 0;
}

#endif