
#include "sandPile.h"

#pragma region 4.3 //OpenMP implementation of the asynchronous version


//OMP_SCHEDULE=static OMP_NUM_THREADS=2 ./run -k asandPile -s 256 -th 256 -tw 64 -v omp_test_2 -wt opt -n
unsigned asandPile_compute_omp_test_2(unsigned nb_iter)
{
  int res = 0;

  for (unsigned it = 1; it <= nb_iter; it++)
  {
    int change = 0;

      //#pragma omp parallel for shared(change)// collapse(2) schedule(runtime) reduction(|:change)
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

//dbg : OMP_NUM_THREADS=16 OMP_SCHEDULE=static ./run -k asandPile -s 32 -tw 8 -th 32 -v omp -wt dbg
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
                    TILE_H - (y == 0) - (y == 0 && TILE_H == DIM));//TILE_H - (y == 0));

        if(TILE_H == DIM) continue;//pas bon iter 1:

        change |=
             do_tile(x + TILE_W, y + TILE_H,
                     TILE_W - (x + 2*TILE_W == DIM), 
                     TILE_H - (y + 2*TILE_H == DIM));
      }

      #pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change)
    for (int y = 0; y < DIM; y += 2*TILE_H)
      for (int x = TILE_W; x < DIM; x += 2*TILE_W){
        //:
        change |=
            do_tile(x, y + (y == 0),
              TILE_W - (x + TILE_W == DIM),
              TILE_H - (y == 0) - (y == 0 && TILE_H == DIM));//etait pas bon iter 7

        if(TILE_H == DIM) continue;//etait pas bon iter 1:

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
                    TILE_H - (y == 0) - (y == 0 && TILE_H == DIM));

        if(TILE_H == DIM) continue;

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
              TILE_H - (y == 0) - (y == 0 && TILE_H == DIM));

        if(TILE_H == DIM) continue;
        
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
  //printf("x=%d y=%d width=%d height=%d\n", x, y, width, height);

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

int asandPile_do_tile_dbg(int x, int y, int width, int height)
{
  //printf("x=%d y=%d width=%d height=%d\n", x, y, width, height);

  int change = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) 
    {
      if(i + 1 >= DIM || j + 1 >= DIM || i - 1 < 0 || j - 1 < 0)
      {
        printf("AH x=%d y=%d width=%d height=%d\n", x, y, width, height);
        return 0;
      }
    }
}

#pragma endregion