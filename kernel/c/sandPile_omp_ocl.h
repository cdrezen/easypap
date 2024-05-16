#include "sandPile.h"

#ifdef ENABLE_OPENCL

#define in_buff cur_buffer
#define out_buff next_buffer

const unsigned int NB_LINES = 2;
double CPU_PERCENT = 0.5;
double cpu_tiles_y = 0;
int _border = 0;

bool solo = false;
unsigned _it = 0;
//TYPE* blank_line;

// cl_mem gpu_change_buff;
// cl_mem zero_buff;
// #define IMAGE 0
// #define BORDER 1
// bool* GPU_CHANGE;

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

  cl_int err = clEnqueueReadBuffer(queue, in_buff, CL_TRUE, _border * DIM * sizeof(unsigned),
                          sizeof(unsigned) * abs(DIM - _border) * DIM, &table(in, _border, 0), 0, NULL, NULL);

  check(err, "Failed to read buffer from GPU");

  ssandPile_refresh_img();
}

// ./run -k ssandPile -g -v ocl_omp -a 0.25

void ssandPile_config_ocl_omp (char *param)
{
  if(!param) return;

  char* endptr;  
  CPU_PERCENT = strtod(param, &endptr); 
  
  if (param == endptr) exit_with_error("Failed to convert arg\n");
}


void ssandPile_init_ocl_omp (void)
{
  ssandPile_init();

  cpu_tiles_y = CPU_PERCENT * NB_TILES_Y;
  _border = cpu_tiles_y * TILE_H;
  //NB_LINES = TILE_H - 1;
  
  printf("init border=%d cpu_tiles_y=%f\n", _border, cpu_tiles_y);

  // GPU_CHANGE = calloc(2, sizeof(bool));
  // blank_line = calloc(DIM, sizeof(unsigned));

  // gpu_change_buff = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof(bool) * 2, NULL, NULL);//CL_MEM_HOST_READ_ONLY
  // if (!gpu_change_buff)
  //   exit_with_error ("Failed to allocate bool buffer");

  // zero_buff = gpu_change_buff;
}

void ssandPile_finalize_ocl_omp (void)
{
  ssandPile_finalize();
  //free(GPU_CHANGE);
  //free(blank_line)
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

struct cpu_result
{
  bool change;
  unsigned nb_copied_lines;
};

int a = 0;

struct cpu_result do_cpu_post(const unsigned border, const size_t offset_read, const TYPE* read_ptr, const size_t read_sz, const size_t offset_write, const unsigned write_y, const size_t write_sz)
{
    unsigned nb_copied_lines = 0;

    //err = clEnqueueReadBuffer (queue, gpu_change_buff, CL_TRUE, 0, sizeof(bool) * 2, GPU_CHANGE, 0, NULL, NULL);
    //check(err, "Failed to read bool from GPU");

    if(!solo)//&& GPU_CHANGE[BORDER]
    {
      //read gpu line at border
      cl_int err = clEnqueueReadBuffer(queue, in_buff, CL_TRUE, offset_read, read_sz, read_ptr, 0, NULL, NULL);
      check(err, "Failed to read buffer from GPU");
    }
    //else printf("solo\n");

    bool change = 0;
    bool cpu_border_changed = false;

      #pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change, cpu_border_changed)
    for(int y = 0; y < border + (!solo); y+=TILE_H) {
      for(int x = 0; x < DIM; x+=TILE_W)
      {
        bool diff = do_tile(x + (x == 0), y + (y == 0),
                     TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                     TILE_H - ((y + TILE_H == DIM) + (y == 0)));

        if (y == border - TILE_H)
        {
          cpu_border_changed |= diff;
        }

        change |= diff;
      }
    }

    if(!solo && cpu_border_changed)//
    {
      //write cpu line(s) at border to gpu
      cl_int err = clEnqueueWriteBuffer (queue, out_buff, CL_TRUE, offset_write, write_sz, &table(out, write_y, 0), 0, NULL, NULL);
      check(err, "Failed to write buffer to GPU");
      nb_copied_lines = NB_LINES;
    }

    // Swap buffers
    {
      cl_mem tmp  = in_buff;
      in_buff = out_buff;
      out_buff = tmp;
      swap_tables();
      solo = !solo;
      //solo = (_it % NB_LINES);
      //gpu_change_buff = zero_buff;   
    }

    return (struct cpu_result){ change, nb_copied_lines };
}


unsigned ssandPile_invoke_ocl_omp (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y}; // global domain size for our calculation
  size_t local[2]  = {TILE_W, TILE_H}; // local domain size for our calculation
  cl_int err;

  const size_t offset_read = _border * DIM * sizeof(unsigned);
  const TYPE* read_ptr = &table(in, _border, 0);
  const size_t read_sz = sizeof (unsigned) * (NB_LINES) * DIM;

  const unsigned write_y = (_border - 1);//
  const size_t offset_write = write_y * DIM * sizeof(unsigned);
  //const TYPE* write_ptr = &table(out, write_y, 0);//border - NB_LINES
  const size_t write_sz = sizeof (unsigned) * (NB_LINES) * DIM;

  unsigned current_border = _border;

  uint64_t clock = monitoring_start_tile(easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {
    _it++;
    //printf("it=%d\n", it);

    //do_cpu_pre();

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &in_buff);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &out_buff);
    err |= clSetKernelArg (compute_kernel, 2, sizeof (unsigned), &current_border);
    //err |= clSetKernelArg (compute_kernel, 3, sizeof (cl_mem), &gpu_change_buff);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local, 0, NULL, NULL);
    check (err, "Failed to execute kernel");

    struct cpu_result result = do_cpu_post(_border, offset_read, read_ptr, read_sz, offset_write, write_y, write_sz);
    
    if(result.change == 0) 
    {
      //if(!GPU_CHANGE[IMAGE])
      {
        clFinish (queue);
        monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));
        return it;
      }
      //else printf("change gpu\n");
    }
    
    current_border = _border - result.nb_copied_lines;

    // bool b = false;
    // err |= clEnqueueFillBuffer(queue, gpu_change_buff, &b, sizeof(bool), 0, sizeof(bool) * 2, 0, NULL, NULL);
    // check(err, "Failed to fill buffer in GPU");
  }

  clFinish (queue);

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  if(do_display)
  {
    err |= clEnqueueWriteBuffer(queue, in_buff, CL_TRUE, 0,
                          sizeof(unsigned) * _border * DIM, &table(in, 0, 0), 0, NULL, NULL);
    check(err, "Failed to write buffer to GPU");
  }

  return 0;
}

#endif