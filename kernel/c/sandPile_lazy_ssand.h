#include "sandPile.h"

#pragma region 4.4 // Lazy OpenMP implementations ssand

static bool* restrict LA_TABLE = NULL;//tableau de booleen pour le LAzy

static inline bool* is_steady_tile(bool *restrict i, int y, int x)
{
  //return i + y * (DIM / TILE_H) + x;
  return i + (y / TILE_H) * (DIM / TILE_H) + (x / TILE_W);
  //return DIM * DIM * step + i + (y / TILE_H) * DIM + x;
}

#define is_steady(y, x) (*is_steady_machin((y), (x)))

bool* is_steady_machin(int y, int x)
{
  
  return is_steady_tile(LA_TABLE, y, x);
}

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

  LA_TABLE = malloc(2 * NB_TILES_X * NB_TILES_Y * sizeof(bool));
  memset(LA_TABLE, 0, 2 * NB_TILES_X * NB_TILES_Y  * sizeof(bool));//met tout à true(1) (pour la première iteration)
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

/* check_steady retourne true si la tuile (x,y) possede des voisins stables 
*  rappel : is_steady retourne true si la tuile est stable
*/
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
  bool firstCheck = true;//

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

        if(diff == 0 && (firstCheck || check_steady(y, x)))//
        {
          #pragma omp atomic
          is_steady(y, x) |= true;

        }
      }
    // print_table();
    
    firstCheck = false;//


    swap_tables();
    if (change == 0)
      return it;
  }

  return 0;
}



#pragma endregion
