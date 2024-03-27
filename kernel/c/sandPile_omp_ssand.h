#include "sandPile.h"

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