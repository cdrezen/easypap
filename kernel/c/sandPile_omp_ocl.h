#include "sandPile.h"

#ifdef ENABLE_OPENCL

#define in_buff cur_buffer
#define out_buff next_buffer

double CPU_GPU_RATIO = 0.5;

double cpu_tiles_y = 0;
int border_top = 0;
int border_bottom = 0;
// double cpu_tiles_x = 0;
// int border_left = 0;
// int border_right = 0;
cl_mem gpu_change_buff;
cl_mem zero_buff;
#define IMAGE 0
#define BORDER 1
bool* GPU_CHANGE;

const unsigned int NB_LINES = 1;


// Only called when --dump or --thumbnails is used
void ssandPile_refresh_img_ocl()
{
  cl_int err = clEnqueueReadBuffer (queue, cur_buffer, CL_TRUE, 0, sizeof (unsigned) * DIM * DIM, TABLE, 0, NULL, NULL);
  check(err, "Failed to read buffer from GPU");

  ssandPile_refresh_img();
}

void ssandPile_refresh_img_ocl_omp()
{
  printf("refresh\n");

  cl_int err = clEnqueueReadBuffer(queue, in_buff, CL_TRUE, border_top * DIM * sizeof(unsigned),
                          sizeof(unsigned) * abs(DIM - border_top) * DIM, &table(in, border_top, 0), 0, NULL, NULL);

  check(err, "Failed to read buffer from GPU");

  ssandPile_refresh_img();
}

// ./run -k ssandPile -g -v ocl_omp -a 0.25

void ssandPile_config_ocl_omp (char *param)
{
  if(!param) return;

  char* endptr;  
  CPU_GPU_RATIO = strtod(param, &endptr); 
  
  if (param == endptr) exit_with_error("Failed to convert arg\n");
}


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
  
  printf("init border_top=%d cpu_tiles_y=%f\n", border_top, cpu_tiles_y);

  GPU_CHANGE = malloc(2 * sizeof(bool));

  gpu_change_buff = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof(bool) * 2, NULL, NULL);//CL_MEM_HOST_READ_ONLY
  if (!gpu_change_buff)
    exit_with_error ("Failed to allocate bool buffer");

  zero_buff = gpu_change_buff;
}

void ssandPile_finalize_ocl_omp (void)
{
  ssandPile_finalize();
  free(GPU_CHANGE);
}

//dbg
int ssandPile_do_tile_dbg(int x, int y, int width, int height)
{
  int diff = 0;
  for (int i = y; i < y + height; i++) {
    for (int j = x; j < x + width; j++)
    {
        //table(in, i, j) = (unsigned int)rand() % 4;
        table(out, i, j) = 1;
        //if(table(in, i, j) != table(out, i, j)) diff = 1;
    }
  }

  return 1;
}

void do_cpu_pre(){}

int do_cpu_post(const size_t offset_gpu, const size_t offset_cpu, const size_t transfer_sz)
{
    cl_int err = 0;

    //err = clEnqueueReadBuffer (queue, gpu_change_buff, CL_TRUE, 0, sizeof(bool) * 2, GPU_CHANGE, 0, NULL, NULL);
    //check(err, "Failed to read bool from GPU");

    //if(GPU_CHANGE[BORDER])
    {
      //read gpu line at border
      err = clEnqueueReadBuffer(queue, in_buff, CL_TRUE, offset_gpu, transfer_sz, &table(in, border_top, 0), 0, NULL, NULL);
      check(err, "Failed to read buffer from GPU");
    }
    // else{
    //   printf("no border change\n");
    //   memcpy(&table(in, border_top, 0), &table(out, border_top, 0), size);
    // }

    int change = 0;
    bool cpu_border_changed = false;


      #pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change)
    for(int y = 0; y < border_top; y+=TILE_H) {
      for(int x = 0; x < DIM; x+=TILE_W)
      {
        int diff = do_tile(x + (x == 0), y + (y == 0),
                     TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                     TILE_H - ((y + TILE_H == DIM) + (y == 0)));

        if (y == border_top - TILE_H){
            cpu_border_changed |= diff;
        }

        change |= diff;
      }
    }

    if(cpu_border_changed)
    {
    //write cpu line at border to gpu
      err = clEnqueueWriteBuffer (queue,  out_buff, CL_TRUE, offset_cpu, transfer_sz, &table(out, border_top - NB_LINES, 0), 0, NULL, NULL);
      check(err, "Failed to write buffer to GPU");
    }

    // Swap buffers
    {
      cl_mem tmp  = in_buff;
      in_buff = out_buff;
      out_buff = tmp;
      swap_tables();
      gpu_change_buff = zero_buff;
    }

    return change;
}


unsigned ssandPile_invoke_ocl_omp (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y}; // global domain size for our calculation
  size_t local[2]  = {TILE_W, TILE_H}; // local domain size for our calculation
  cl_int err;

  const size_t offset_gpu = border_top * DIM * sizeof(unsigned);
  const size_t offset_cpu = (border_top - NB_LINES) * DIM * sizeof(unsigned);
  const size_t transfer_sz = sizeof (unsigned) * (NB_LINES) * DIM;


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
    err |= clSetKernelArg (compute_kernel, 3, sizeof (cl_mem), &gpu_change_buff);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local, 0, NULL, NULL);
    check (err, "Failed to execute kernel");

    int change = do_cpu_post(offset_gpu, offset_cpu, transfer_sz);//, read_dest, write_source);
    if(change == 0) 
    {
      //if(!GPU_CHANGE[IMAGE])
      {
        clFinish (queue);
        monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));
        return it;
      }
      //else printf("change gpu\n");
    } 

    // bool b = false;
    // err |= clEnqueueFillBuffer(queue, gpu_change_buff, &b, sizeof(bool), 0, sizeof(bool) * 2, 0, NULL, NULL);
    // check(err, "Failed to fill buffer in GPU");
  }

  clFinish (queue);

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  if(do_display)
  {
    err |= clEnqueueWriteBuffer(queue, in_buff, CL_TRUE, 0,
                          sizeof(unsigned) * border_top * DIM, &table(in, 0, 0), 0, NULL, NULL);
    check(err, "Failed to write buffer to GPU");
  }

  return 0;
}

#endif