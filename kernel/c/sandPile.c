#include "easypap.h"

#include <omp.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <unistd.h>

typedef unsigned int TYPE;

static TYPE* restrict TABLE = NULL;
//static TYPE* TABLE = NULL;

static inline TYPE *atable_cell(TYPE *restrict i, int y, int x)
{
  return i + y * DIM + x;
}

#define atable(y, x) (*atable_cell(TABLE, (y), (x)))

static inline TYPE *table_cell(TYPE *restrict i, int step, int y, int x)
{
  return DIM * DIM * step + i + y * DIM + x;
}

#define table(step, y, x) (*table_cell(TABLE, (step), (y), (x)))

static int in = 0;
static int out = 1;

static inline void swap_tables()
{
  int tmp = in;
  in = out;
  out = tmp;
}

#define RGB(r, g, b) rgba(r, g, b, 0xFF)

static TYPE max_grains;

void asandPile_refresh_img()
{
  unsigned long int max = 0;
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
    {
      int g = table(in, i, j);
      int r, v, b;
      r = v = b = 0;
      if (g == 1)
        v = 255;
      else if (g == 2)
        b = 255;
      else if (g == 3)
        r = 255;
      else if (g == 4)
        r = v = b = 255;
      else if (g > 4)
        r = b = 255 - (240 * ((double)g) / (double)max_grains);

      cur_img(i, j) = RGB(r, v, b);
      if (g > max)
        max = g;
    }
  max_grains = max;
}

/////////////////////////////  Initial Configurations

static inline void set_cell(int y, int x, unsigned v)
{
  atable(y, x) = v;
  if (gpu_used)
    cur_img(y, x) = v;
}

void asandPile_draw_4partout(void);

void asandPile_draw(char *param)
{
  // Call function ${kernel}_draw_${param}, or default function (second
  // parameter) if symbol not found
  hooks_draw_helper(param, asandPile_draw_4partout);
}

void ssandPile_draw(char *param)
{
  hooks_draw_helper(param, asandPile_draw_4partout);
}

void asandPile_draw_4partout(void)
{
  max_grains = 8;
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      set_cell(i, j, 4);
}

void asandPile_draw_DIM(void)
{
  max_grains = DIM;
  for (int i = DIM / 4; i < DIM - 1; i += DIM / 4)
    for (int j = DIM / 4; j < DIM - 1; j += DIM / 4)
      set_cell(i, j, i * j / 4);
}

void asandPile_draw_alea(void)
{
  max_grains = 5000;
  for (int i = 0; i < DIM >> 3; i++)
  {
    set_cell(1 + random() % (DIM - 2), 1 + random() % (DIM - 2), 1000 + (random() % (4000)));
  }
}

void asandPile_draw_big(void)
{
  const int i = DIM / 2;
  set_cell(i, i, 100000);
}

static void one_spiral(int x, int y, int step, int turns)
{
  int i = x, j = y, t;

  for (t = 1; t <= turns; t++)
  {
    for (; i < x + t * step; i++)
      set_cell(i, j, 3);
    for (; j < y + t * step + 1; j++)
      set_cell(i, j, 3);
    for (; i > x - t * step - 1; i--)
      set_cell(i, j, 3);
    for (; j > y - t * step - 1; j--)
      set_cell(i, j, 3);
  }
  set_cell(i, j, 4);

  for (int i = -2; i < 3; i++)
    for (int j = -2; j < 3; j++)
      set_cell(i + x, j + y, 3);
}

static void many_spirals(int xdebut, int xfin, int ydebut, int yfin, int step,
                         int turns)
{
  int i, j;
  int size = turns * step + 2;

  for (i = xdebut + size; i < xfin - size; i += 2 * size)
    for (j = ydebut + size; j < yfin - size; j += 2 * size)
      one_spiral(i, j, step, turns);
}

static void spiral(unsigned twists)
{
  many_spirals(1, DIM - 2, 1, DIM - 2, 2, twists);
}

void asandPile_draw_spirals(void)
{
  spiral(DIM / 32);
}

// shared functions

#define ALIAS(fun)       \
  void ssandPile_##fun() \
  {                      \
    asandPile_##fun();   \
  }

ALIAS(refresh_img);
ALIAS(draw_4partout);
ALIAS(draw_DIM);
ALIAS(draw_alea);
ALIAS(draw_big);
ALIAS(draw_spirals);

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Synchronous Kernel
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

void ssandPile_init()
{
  TABLE = calloc(2 * DIM * DIM, sizeof(TYPE));
}

void ssandPile_finalize()
{
  free(TABLE);
}

int ssandPile_do_tile_default(int x, int y, int width, int height)
{
  int diff = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
    {
      table(out, i, j) = table(in, i, j) % 4;
      table(out, i, j) += table(in, i + 1, j) / 4;
      table(out, i, j) += table(in, i - 1, j) / 4;
      table(out, i, j) += table(in, i, j + 1) / 4;
      table(out, i, j) += table(in, i, j - 1) / 4;
      if (table(out, i, j) >= 4)
        diff = 1;
    }

  return diff;
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned ssandPile_compute_seq(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    int change = do_tile(1, 1, DIM - 2, DIM - 2);
    swap_tables();
    if (change == 0)
      return it;
  }
  return 0;
}

unsigned ssandPile_compute_tiled(unsigned nb_iter)
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

// Only called when --dump or --thumbnails is used
void ssandPile_refresh_img_ocl()
{
  cl_int err;

  err =
      clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0,
                          sizeof(unsigned) * DIM * DIM, TABLE, 0, NULL, NULL);
  check(err, "Failed to read buffer from GPU");

  ssandPile_refresh_img();
}

#endif

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Asynchronous Kernel
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

void asandPile_init()
{
  in = out = 0;
  if (TABLE == NULL)
  {
    const unsigned size = DIM * DIM * sizeof(TYPE);

    PRINT_DEBUG('u', "Memory footprint = 2 x %d bytes\n", size);

    TABLE = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
}

void asandPile_finalize()
{
  const unsigned size = DIM * DIM * sizeof(TYPE);

  munmap(TABLE, size);
}

///////////////////////////// Version séquentielle simple (seq)
// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0

int asandPile_do_tile_default(int x, int y, int width, int height)
{
  int change = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      if (atable(i, j) >= 4)
      {
        atable(i, j - 1) += atable(i, j) / 4;
        atable(i, j + 1) += atable(i, j) / 4;
        atable(i - 1, j) += atable(i, j) / 4;
        atable(i + 1, j) += atable(i, j) / 4;
        atable(i, j) %= 4;
        change = 1;
      }
  return change;
}

unsigned asandPile_compute_seq(unsigned nb_iter)
{
  int change = 0;
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    // On traite toute l'image en un coup (oui, c'est une grosse tuile)
    change = do_tile(1, 1, DIM - 2, DIM - 2);

    if (change == 0)
      return it;
  }
  return 0;
}

unsigned asandPile_compute_tiled(unsigned nb_iter)
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
    if (change == 0)
      return it;
  }

  return 0;
}

#pragma region 4.1 // ILP optimization

#pragma GCC optimize ("unroll-loops")
int ssandPile_do_tile_opt(int x, int y, int width, int height)
{
  int diff = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
    {
      TYPE *restrict cell_in = table_cell(TABLE, in, i, j);
      TYPE *restrict cell_out = table_cell(TABLE, out, i, j);

      *cell_out = table(in, i, j) % 4
                 + (table(in, i, j - 1) / 4) 
                 + (table(in, i, j + 1) / 4) 
                 + (*(cell_in - DIM) / 4)  // table(in, i - 1, j) / 4
                 + (*(cell_in + DIM) / 4); // table(in, i + 1, j) / 4
      
      if (*cell_out >= 4)
        diff = 1;      
    }

  return diff;
}

#pragma GCC optimize ("unroll-loops")
int asandPile_do_tile_opt(int x, int y, int width, int height)
{
  int change = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
    {
      TYPE *restrict cell = atable_cell(TABLE, i, j);
      if (*cell >= 4)
      {
        const TYPE cell_quarter = *cell >> 2;// /4

        atable(i, j - 1) += cell_quarter;
        atable(i, j + 1) += cell_quarter;
        *(cell - DIM) += cell_quarter; // atable(i - 1, j)
        *(cell + DIM) += cell_quarter; // atable(i + 1, j) 
        *cell &= 3;//%4

        change = 1;
      }
    }
  
  return change;
}

#pragma endregion

#pragma region 4.2 // OpenMP implementation of the synchronous version

unsigned ssandPile_compute_omp(unsigned nb_iter)
{
  int res = 0;

  for (unsigned it = 1; it <= nb_iter; it++)
  {
    int change = 0;
      // On traite toute l'image en un coup
    int x = 1; int y = 1; int width = DIM - 2; int height = DIM - 2;

    #pragma omp parallel for collapse(2) shared(change) schedule(runtime)//reduction(+:cell_out[2* DIM * DIM])
    for (int i = y; i < y + height; i++)
      for (int j = x; j < x + width; j++)
      {
        TYPE *restrict cell_in = table_cell(TABLE, in, i, j);
        TYPE *restrict cell_out = table_cell(TABLE, out, i, j);

        const TYPE calc = *cell_in % 4 + (table(in, i, j - 1) / 4) + (table(in, i, j + 1) / 4) + (*(cell_in - DIM) / 4)  + (*(cell_in + DIM) / 4);

        //#pragma omp atomic write
        *cell_out = calc;
        
        if (*cell_out >= 4)
          //#pragma omp atomic write
          change = 1;      
      }

    swap_tables();

    if (change == 0)
      {
        res = it;
        break;
      }
  }

  return res;
}


unsigned ssandPile_compute_omp_tiled(unsigned nb_iter)
{
  int res = 0;
  int change = 0;

  for (unsigned it = 1; it <= nb_iter; it++)
  {
    change = 0;

    #pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change)
    for (int y = 0; y < DIM; y += TILE_H)
    {
      for (int x = 0; x < DIM; x += TILE_W)
      {
        change |= do_tile(x + (x == 0), y + (y == 0),
                           TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                           TILE_H - ((y + TILE_H == DIM) + (y == 0)));
      }
    }

    swap_tables();

    if (change == 0)
    {
      res = it;
      break;
    }
  }

  return res;
}

unsigned ssandPile_compute_omp_taskloop(unsigned nb_iter)
{
  int res = 0;
  int change = 0;

  #pragma omp parallel master firstprivate(change)
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    change = 0;

    #pragma omp taskloop collapse(2) grainsize(4)  shared(change)
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        change |=
            do_tile(x + (x == 0), y + (y == 0),
                    TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                    TILE_H - ((y + TILE_H == DIM) + (y == 0)));

    swap_tables();

    if (change == 0)
    {
      res = it;
      break;
    }
  }

  return res;
}


unsigned ssandPile_compute_omp_task(unsigned nb_iter)
{
  int res = 0;
  unsigned int A[10][10];

  #pragma omp parallel master //shared(change)
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    int change = 0;

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)               //depend(out : A[y][x]) depend(in : A[y-1][x], A[y][x-1])
        #pragma omp task firstprivate(x,y) shared(change) depend(out : A[y][x]) depend(in : A[y-1][x], A[y][x-1], A[y][x+1], A[y][x-1])//depend(out : table(out, y, x)) depend(in : table(in, y, x), table(in, y, x-1), table(in, y, x+1), table(in, y-1, x), table(in, y+1, x))//depend(out : A[y][x]) depend(in : A[y][x], A[y-1][x], A[y+1][x], A[y][x-1], A[y][x+1])
        #pragma omp atomic
        change |=
            do_tile(x + (x == 0), y + (y == 0),
                    TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                    TILE_H - ((y + TILE_H == DIM) + (y == 0)));

    #pragma omp taskwait

    swap_tables();

    if (change == 0)
    {
      res = it;
      break;
    }
  }

  return res;
}

#pragma endregion

#pragma region 4.3 //OpenMP implementation of the asynchronous version

static void touch_tile (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      next_img (i, j) = cur_img (j, i) = next_img (j, i) = cur_img (i, j) = 0; //atable_cell =?
}

unsigned asandPile_ft (unsigned nb_iter)
{
      #pragma omp parallel for collapse(2) schedule(runtime)
    for (int y = 0; y < DIM; y += 2*TILE_H)
      for (int x = 0; x < DIM; x += 2*TILE_W)
      {
        touch_tile(x + (x == 0), y + (y == 0),
                    TILE_W - (x == 0),
                    TILE_H - (y == 0));
        touch_tile(x + TILE_W, y + TILE_H,
                    TILE_W - (x + 2*TILE_W == DIM), 
                    TILE_H - (y + 2*TILE_H == DIM));
      }

      #pragma omp parallel for collapse(2) schedule(runtime)
    for (int y = 0; y < DIM; y += 2*TILE_H)
      for (int x = TILE_W; x < DIM; x += 2*TILE_W){

        touch_tile(x, y + (y == 0),
                    TILE_W - (x + TILE_W == DIM),
                    TILE_H - (y == 0));

        touch_tile(x - TILE_W + (x == TILE_W), y + TILE_H,
                  TILE_W - (x == TILE_W),
                  TILE_H - (y + 2*TILE_H == DIM));
  }

  return 0;
}


//OMP_SCHEDULE=static OMP_NUM_THREADS=2 ./run -k asandPile -s 256 -th 256 -tw 64 -v omp_test_2 -wt opt -n
unsigned asandPile_compute_omp_test_2(unsigned nb_iter)
{
  int res = 0;

  for (unsigned it = 1; it <= nb_iter; it++)
  {
    int change = 0;

      //#pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change)
    for (int y = 0; y < DIM; y += TILE_H) {
        #pragma omp parallel for shared(change) //reduction(|:change)
      for (int x = 0; x < DIM; x += TILE_W)
      {
        if(((x / TILE_W) + (y /TILE_H)) % 2) continue;

        //#pragma omp atomic
        change |=
            do_tile(x + (x == 0), y + (y == 0),
                    TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                    TILE_H - ((y + TILE_H == DIM) + (y == 0)));
      }
        #pragma omp parallel for shared(change) //reduction(|:change)
      for (int x = 0; x < DIM; x += TILE_W)
      {
        if(((x / TILE_W) + (y /TILE_H)) % 2 == 0) continue;

        //#pragma omp atomic
        change |=
            do_tile(x + (x == 0), y + (y == 0),
                    TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                    TILE_H - ((y + TILE_H == DIM) + (y == 0)));
      } 
    }

    //#pragma omp barrier

    if (change == 0)
    {
      res = it;
      break;
    }
  }

  return res;
}

unsigned asandPile_compute_omp(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    int change = 0;
#pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change)
    for (int y = 0; y < DIM; y += 2*TILE_H)
      for (int x = 0; x < DIM; x += 2*TILE_W){
        change |=
            do_tile(x + (x == 0), y + (y == 0),
                    TILE_W - (x == 0),
                    TILE_H - (y == 0));
        change |=
             do_tile(x + TILE_W, y + TILE_H,
                     TILE_W - (x + 2*TILE_W == DIM), 
                     TILE_H - (y + 2*TILE_H == DIM));
      }

#pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change)
    for (int y = 0; y < DIM; y += 2*TILE_H)
      for (int x = TILE_W; x < DIM; x += 2*TILE_W){

        change |=
            do_tile(x, y + (y == 0),
              TILE_W - (x + TILE_W == DIM),
              TILE_H - (y == 0));

        change |=
            do_tile(x - TILE_W + (x == TILE_W), y + TILE_H,
              TILE_W - (x == TILE_W),
              TILE_H - (y + 2*TILE_H == DIM));
      }

    if (change == 0)
      return it;
  }

  return 0;
}

unsigned asandPile_compute_omp_task(unsigned nb_iter)
{
  int res = 0;

  #pragma omp parallel master
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    int A[10][10];
    int change = 0;
    for (int y = 0; y < DIM; y += 2*TILE_H)
      for (int x = 0; x < DIM; x += 2*TILE_W)
      {
        #pragma omp task shared(change) firstprivate(x,y) depend(out : A[y][x]) depend(in : A[y-1][x], A[y][x-1], A[y][x+1], A[y][x-1])
        change |=
            do_tile(x + (x == 0), y + (y == 0),
                    TILE_W - (x == 0),
                    TILE_H - (y == 0));

        #pragma omp task shared(change) firstprivate(x,y) depend(out : A[y][x]) depend(in : A[y-1][x], A[y][x-1], A[y][x+1], A[y][x-1])
        change |=
             do_tile(x + TILE_W, y + TILE_H,
                     TILE_W - (x + 2*TILE_W == DIM), 
                     TILE_H - (y + 2*TILE_H == DIM));

        #pragma omp taskwait

        int x1 = TILE_W + x;
        
        #pragma omp task shared(change) firstprivate(x1,y) depend(out : A[y][x]) depend(in : A[y-1][x], A[y][x-1], A[y][x+1], A[y][x-1])
        change |=
            do_tile(x1, y + (y == 0),
              TILE_W - (x1 + TILE_W == DIM),
              TILE_H - (y == 0));
        
        #pragma omp task shared(change) firstprivate(x1,y) depend(out : A[y][x]) depend(in : A[y-1][x], A[y][x-1], A[y][x+1], A[y][x-1])
        change |=
            do_tile(x1 - TILE_W + (x1 == TILE_W), y + TILE_H,
              TILE_W - (x1 == TILE_W),
              TILE_H - (y + 2*TILE_H == DIM));
      }

    #pragma omp taskwait

    if (change == 0)
    {
      res = it;
      break;
    }
  }

  return res;
}

// fonction inline pour génerer le code du calcul des cellules voisines qui necessitent une synchronisation si elles sont en bord de tuile
static inline void asandPile_do_neighbor_cell(TYPE* cell, TYPE val, int ij, int border_pos)
{
    if(ij == border_pos)
    {
      #pragma omp atomic
      *cell += val;
    }
    else 
      *cell += val;
}

#pragma GCC optimize ("unroll-loops")
int asandPile_do_tile_opt1(int x, int y, int width, int height)
{
  int change = 0;
  // const bool sync_top = y > 0;
  // const bool sync_left = x > 0;
  // const bool sync_right = x + width < DIM;
  // const bool sync_bottom = y + height < DIM;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) { 
      TYPE *restrict cell = atable_cell(TABLE, i, j);
      if (*cell >= 4)
      {
        const TYPE cell_quarter = *cell >> 2;// /4
        
        asandPile_do_neighbor_cell(cell-DIM, cell_quarter, i, y);
        asandPile_do_neighbor_cell(cell-1, cell_quarter, j, x);
        asandPile_do_neighbor_cell(cell+1, cell_quarter, j, x + width - 1);
        asandPile_do_neighbor_cell(cell+DIM, cell_quarter, i, y + height - 1);

        *cell &= 3;
          
        change = 1;
      }
    }
  return change;
}

#pragma endregion

#pragma region 4.4 // Lazy OpenMP implementations ssand

static bool* restrict LA_TABLE = NULL;//tableau de booleen pour le LAzy

static inline bool* is_steady_tile(bool *restrict i, int y, int x)
{
  //return i + y * (DIM / TILE_H) + x;
  return i + (y / TILE_H) * (DIM / TILE_H) + (x / TILE_W);
  //return DIM * DIM * step + i + (y / TILE_H) * DIM + x;
}

#define is_steady(y, x) (*is_steady_tile(LA_TABLE, (y), (x)))

void print_table()
{
  for (int y = 0; y < DIM; y+=TILE_H)
  {
      for (int x = 0; x < DIM; x+=TILE_W)
      {
        printf("%d ", is_steady(y, x));
      }
      printf("\n");
  }
}

void ssandPile_init_lazy()
{
  ssandPile_init();

  LA_TABLE = malloc((DIM / TILE_H) * (DIM / TILE_W) * sizeof(bool));
  memset(LA_TABLE, 0, (DIM / TILE_H) * (DIM / TILE_W) * sizeof(bool));//met tout à true(1) (pour la première iteration)
  // print_table();
}

void ssandPile_finish_lazy()
{
  ssandPile_finalize();

  free(LA_TABLE);
}

//   // const bool sync_top = y > 0;
//   // const bool sync_left = x > 0;
//   // const bool sync_right = x + width < DIM;
//   // const bool sync_bottom = y + height < DIM

#pragma GCC optimize ("unroll-loops")
int ssandPile_do_tile_lazy(int x, int y, int width, int height)
{
  int diff = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
    {
      TYPE *restrict cell_in = table_cell(TABLE, in, i, j);
      TYPE *restrict cell_out = table_cell(TABLE, out, i, j);

      *cell_out = table(in, i, j) % 4
                 + (table(in, i, j - 1) / 4) 
                 + (table(in, i, j + 1) / 4) 
                 + (*(cell_in - DIM) / 4)  // table(in, i - 1, j) / 4
                 + (*(cell_in + DIM) / 4); // table(in, i + 1, j) / 4
      
      if (*cell_out >= 4)
      {
        diff = 1;
        if(j == (x + width - 1) && width != (TILE_W - 1)) 
            is_steady(y, x + TILE_W) = false;
        if(j == x && width != (TILE_W - 1)) 
            is_steady(y, x - TILE_W) = false;
        if(i == (y + height - 1) && height != (TILE_H - 1)) 
            is_steady(y + TILE_H, x) = false;
        if(i == y && height != (TILE_H - 1)) 
            is_steady(y - TILE_H, x) = false;
      }   
    }

  return diff;
}

// fonction inline pour génerer le code du calcul des cellules voisines qui necessitent une synchronisation si elles sont en bord de tuile
// static inline void check_steady(int xy, int border, int y, int x)
// {
//     if(xy == border)
//     {
      
//     }
// }

bool check_steady(int y, int x)
{
  if(x == 0 && y != 0 && y != DIM - TILE_H){
    return is_steady(y + TILE_H, x) && is_steady(y - TILE_H, x) && is_steady(y, x + TILE_W);
  }
  if(y == 0 && x != 0 && x != DIM - TILE_W){
    return is_steady(y + TILE_H, x) && is_steady(y, x - TILE_W) && is_steady(y, x + TILE_W);
  }
  if(x == DIM - TILE_W && y != 0 && y != DIM - TILE_H){
    return is_steady(y + TILE_H, x) && is_steady(y - TILE_H, x) && is_steady(y, x - TILE_W);
  }
  if(y == DIM - TILE_H && x != 0 && x != DIM - TILE_W){
    return is_steady(y - TILE_H, x) && is_steady(y, x + TILE_W) && is_steady(y, x - TILE_W);
  }

  if(y == 0 && x == 0){
    return is_steady(y + TILE_H, x) && is_steady(y, x + TILE_W);
  }
  if(y == 0 && x == DIM - TILE_W){
    return is_steady(y + TILE_H, x) && is_steady(y, x - TILE_W);
  }
  if(y == DIM - TILE_H && x == 0){
    return is_steady(y - TILE_H, x) && is_steady(y, x + TILE_W);
  }
  if(y == DIM - TILE_H && x == DIM - TILE_W){
    return is_steady(y - TILE_H, x) && is_steady(y, x - TILE_W);
  }

  return is_steady(y - TILE_H, x) && is_steady(y + TILE_H, x) && is_steady(y, x - TILE_W) && is_steady(y, x + TILE_W);
}

unsigned ssandPile_compute_lazy(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    int change = 0;
  #pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change)
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
      {
        if(is_steady(y, x)) continue;

        int diff = do_tile(x + (x == 0), y + (y == 0),
                    TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                    TILE_H - ((y + TILE_H == DIM) + (y == 0)));

        change |= diff;

        if(diff == 0 && check_steady(y, x))
        {
          
          #pragma omp atomic
          is_steady(y, x) |= true;
        }
      }
    // print_table();


    swap_tables();
    if (change == 0)
      return it;
  }

  return 0;
}



#pragma endregion


#pragma region 4.4 // Lazy OpenMP implementations asand

static inline bool* is_active_cell(bool *restrict i, int y, int x)
{
  return i + y * (DIM / TILE_H) + x;
}

#define is_active(y, x) (*is_active_cell(LA_TABLE, (y), (x)))

void asandPile_init_lazy()
{
  asandPile_init();

  LA_TABLE = malloc((DIM / TILE_H) * (DIM / TILE_W) * sizeof(bool));
  memset(LA_TABLE, 1, (DIM / TILE_H) * (DIM / TILE_W) * sizeof(bool));//met tout à true(1) (pour la première iteration)
}

void asandPile_finish_lazy()
{
  asandPile_finalize();

  free(LA_TABLE);
}

static inline void asandPile_lazy_sync(bool diff, int x, int y)
{
  if(diff)
  {
    #pragma omp atomic
    is_active(y, x) |= (bool)diff;

    #pragma omp atomic
    is_active(y, (x - 1 + (x == 0))) |= true;
    #pragma omp atomic
    is_active(y, (x + 1 - (x + TILE_W == DIM))) |= true;
    #pragma omp atomic
    is_active((y - 1 + (y == 0)), x) |= true;
    #pragma omp atomic
    is_active((y + 1 - (y + TILE_H == DIM)), x) |= true;
  }
  else
  {
    #pragma omp atomic
    is_active(y, x) &= (bool)diff;
  }
}

// fonction inline pour génerer le code du calcul des cellules voisines qui necessitent une synchronisation si elles sont en bord de tuile
static inline void asandPile_do_neighbor_cell_lazy(TYPE* cell, TYPE val, int ij, int border_pos, bool* is_active_tile)
{
    if(ij == border_pos)
    {
      #pragma omp atomic
      *cell += val;

      #pragma omp atomic
      *is_active_tile |= true;
    }
    else 
      *cell += val;
}

#pragma GCC optimize ("unroll-loops")
int asandPile_do_tile_oo1(int x, int y, int width, int height)
{
  int change = 0;
  // const bool sync_top = y > 0;
  // const bool sync_left = x > 0;
  // const bool sync_right = x + width < DIM;
  // const bool sync_bottom = y + height < DIM;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) { 
      TYPE *restrict cell = atable_cell(TABLE, i, j);
      if (*cell >= 4)
      {
        const TYPE cell_quarter = *cell >> 2;// /4

        bool *restrict is_active_tile = is_active_cell(LA_TABLE, y / TILE_H, x / TILE_W);
        
        asandPile_do_neighbor_cell_lazy(cell-DIM, cell_quarter, i, y, is_active_tile + (DIM/TILE_H));
        asandPile_do_neighbor_cell_lazy(cell-1, cell_quarter, j, x, is_active_tile - 1);
        asandPile_do_neighbor_cell_lazy(cell+1, cell_quarter, j, x + width - 1, is_active_tile + 1);
        asandPile_do_neighbor_cell_lazy(cell+DIM, cell_quarter, i, y + height - 1, is_active_tile + (DIM/TILE_H));

        *cell &= 3;
          
        change = 1;
      }
    }
  return change;
}

unsigned asandPile_compute_lazy(unsigned nb_iter)
{
  int res = 0;

  for (unsigned it = 1; it <= nb_iter; it++)
  {
    int change = 0;
#pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change)
    for (int y = 0; y < DIM; y += 2*TILE_H)
      for (int x = 0; x < DIM; x += 2*TILE_W){

        bool last = false;
        #pragma omp atomic
        last |= is_active(y/TILE_H, x/TILE_W);

        if(!last) continue;

        change |=
            do_tile(x + (x == 0), y + (y == 0),
                    TILE_W - (x == 0),
                    TILE_H - (y == 0));
        change |=
             do_tile(x + TILE_W, y + TILE_H,
                     TILE_W - (x + 2*TILE_W == DIM), 
                     TILE_H - (y + 2*TILE_H == DIM));

        asandPile_lazy_sync((bool)change, x/TILE_W, y/TILE_H);
      }

#pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change)
    for (int y = 0; y < DIM; y += 2*TILE_H)
      for (int x = TILE_W; x < DIM; x += 2*TILE_W){

        bool last = false;
        #pragma omp atomic
        last |= is_active(y/TILE_H, x/TILE_W);

        if(!last) continue;

        change |=
            do_tile(x, y + (y == 0),
              TILE_W - (x + TILE_W == DIM),
              TILE_H - (y == 0));

        change |=
            do_tile(x - TILE_W + (x == TILE_W), y + TILE_H,
              TILE_W - (x == TILE_W),
              TILE_H - (y + 2*TILE_H == DIM));

        asandPile_lazy_sync((bool)change, x/TILE_W, y/TILE_H);
      }

    if (change == 0)
    {
      bool is_done = true;

      for (int y = 0; y < DIM/TILE_H; y++)
        for (int x = 0; x < DIM/TILE_W; x++){
          if(is_active(y, x)) 
          {
            is_done = false;
            break;
          }
        }
      printf("not done it=%d\n", it);
      if(!is_done) continue;

      res = it;
      break;
    }
  }

  return res;
}




#pragma endregion