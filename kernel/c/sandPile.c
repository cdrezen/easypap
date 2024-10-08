#include "sandPile.h"
#include "sandPile_omp_ssand.h"
#include "sandPile_omp_asand.h"
#include "sandPile_lazy_ssand.h"
#include "sandPile_lazy_asand.h"
#include "sandPile_avx.h"
#include "sandPile_omp_ocl.h"

static inline TYPE *atable_cell(TYPE *restrict i, int y, int x)
{
  return i + y * DIM + x;
}

static inline TYPE *table_cell(TYPE *restrict i, int step, int y, int x)
{
  return DIM * DIM * step + i + y * DIM + x;
}

static inline void swap_tables()
{
  int tmp = in;
  in = out;
  out = tmp;
}

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

static void ssandPile_touch_tile (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      table(in, i, j) = table(in, j, i) = table(out, j, i) = table(out, i, j) = 0;
}

static void asandPile_touch_tile (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      atable(i, j) = atable(j, i) = 0;
}

#pragma GCC optimize ("unroll-loops")
unsigned ssandPile_ft(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    #pragma omp parallel for collapse(2) schedule(runtime)
    for (int y = 0; y < DIM; y += TILE_H)
    {
      for (int x = 0; x < DIM; x += TILE_W)
      {
        ssandPile_touch_tile(x + (x == 0), y + (y == 0),
                           TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                           TILE_H - ((y + TILE_H == DIM) + (y == 0)));
      }
    }

    swap_tables();
  }

  return 0;
}

//OMP_NUM_THREADS=42 OMP_SCHEDULE=static ./run -k asandPile -s 512  -tw 4 -th 512 -v omp -wt opt1 -n -du -sh -ft
unsigned asandPile_ft (unsigned nb_iter)
{
      #pragma omp parallel for collapse(2) schedule(runtime)
    for (int y = 0; y < DIM; y += 2*TILE_H)
      for (int x = 0; x < DIM; x += 2*TILE_W)
      {
        asandPile_touch_tile(x + (x == 0), y + (y == 0),
                    TILE_W - (x == 0),
                    TILE_H - (y == 0) - (y == 0 && TILE_H == DIM));//TILE_H - (y == 0));

        if(TILE_H == DIM) continue;
        
        asandPile_touch_tile(x + TILE_W, y + TILE_H,
                    TILE_W - (x + 2*TILE_W == DIM), 
                    TILE_H - (y + 2*TILE_H == DIM));
      }

      #pragma omp parallel for collapse(2) schedule(runtime)
    for (int y = 0; y < DIM; y += 2*TILE_H)
      for (int x = TILE_W; x < DIM; x += 2*TILE_W){

        asandPile_touch_tile(x, y + (y == 0),
                    TILE_W - (x + TILE_W == DIM),
                    TILE_H - (y == 0) - (y == 0 && TILE_H == DIM));//TILE_H - (y == 0));

        if(TILE_H == DIM) continue;

        asandPile_touch_tile(x - TILE_W + (x == TILE_W), y + TILE_H,
                  TILE_W - (x == TILE_W),
                  TILE_H - (y + 2*TILE_H == DIM));
  }

  return 0;
}

#pragma endregion

#pragma region 4.5

//ssandPile_do_tile_avx -> sandPile_avx.h

#pragma endregion





