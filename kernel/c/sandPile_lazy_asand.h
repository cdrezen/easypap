#include "sandPile.h"
#pragma region 4.4 // Lazy OpenMP implementations asand

void asandPile_init_lazy()
{
  asandPile_init();
  LA_TABLE = calloc(2 * NB_TILES_X * NB_TILES_Y, sizeof(bool));
}

void asandPile_finish_lazy()
{
  asandPile_finalize();
  free(LA_TABLE);
}

static int inl = 0;
static int outl = 1;

static inline void swap_latable()
{
  int tmp = inl;
  inl = outl;
  outl = tmp;
}

#pragma GCC optimize ("unroll-loops")
int asandPile_do_tile_lazy(int x, int y, int width, int height)
{
  int change = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) 
    {
      TYPE* cell = &atable(i, j);
      TYPE last_cell = *cell;

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

void asandPile_process_lazy(int diff, int diff1, int x, int y)
{
  if(diff == 0 && diff1 == 0) is_steady(inl, y, x) |= true;
  else
  {
    is_steady(outl, y, x) &= false;

    if (!(x == 0))
      #pragma omp atomic
      is_steady(outl, y, x - TILE_W) &= false;

    if (!(x + TILE_W == DIM))
      #pragma omp atomic
      is_steady(outl, y, x + TILE_W) &= false;

    if (!(y + TILE_H == DIM))
      #pragma omp atomic
      is_steady(outl, y + TILE_H, x) &= false;

    if (!(y == 0))
      #pragma omp atomic
      is_steady(outl, y - TILE_H, x) &= false;
  }
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

        if(is_steady(inl, y, x) && is_steady(outl, y, x)) continue;

        int diff = do_tile(x + (x == 0), y + (y == 0),
                    TILE_W - (x == 0),
                    TILE_H - (y == 0));
            
        int diff1 = do_tile(x + TILE_W, y + TILE_H,
                     TILE_W - (x + 2*TILE_W == DIM), 
                     TILE_H - (y + 2*TILE_H == DIM));

        change |= diff || diff1;

        asandPile_process_lazy(diff, diff1, x, y);
      }

      #pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change)
    for (int y = 0; y < DIM; y += 2*TILE_H)
      for (int x = TILE_W; x < DIM; x += 2*TILE_W){

        if(is_steady(inl, y, x) && is_steady(outl, y, x)) continue;

        int diff = do_tile(x, y + (y == 0),
                    TILE_W - (x + TILE_W == DIM),
                    TILE_H - (y == 0));
            
        int diff1 = do_tile(x - TILE_W + (x == TILE_W), y + TILE_H,
              TILE_W - (x == TILE_W),
              TILE_H - (y + 2*TILE_H == DIM));
        
        change |= diff || diff1;

        asandPile_process_lazy(diff, diff1, x, y);
      }
    
    swap_latable();
    if (change == 0) return it;
  }

  return res;
}




#pragma endregion