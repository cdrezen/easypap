#include "sandPile.h"
#pragma region 4.4 // Lazy OpenMP implementations asand

static bool* restrict LAA_TABLE = NULL;//tableau de booleen pour le LAzy Async

static inline bool* is_active_cell(bool *restrict i, int y, int x)
{
  return i + y * (DIM / TILE_H) + x;
}

#define is_active(y, x) (*is_active_cell(LAA_TABLE, (y), (x)))

void asandPile_init_lazy()
{
  asandPile_init();

  LAA_TABLE = malloc((DIM / TILE_H) * (DIM / TILE_W) * sizeof(bool));
  memset(LAA_TABLE, 1, (DIM / TILE_H) * (DIM / TILE_W) * sizeof(bool));//met tout à true(1) (pour la première iteration)
}

void asandPile_finish_lazy()
{
  asandPile_finalize();

  free(LAA_TABLE);
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

        bool *restrict is_active_tile = is_active_cell(LAA_TABLE, y / TILE_H, x / TILE_W);
        
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