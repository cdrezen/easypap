#include "sandPile.h"

#pragma region 4.4 // Lazy OpenMP implementations ssand

static bool* restrict LA_TABLE = NULL;//tableau de booleen pour le LAzy

static inline bool* is_steady_tile(bool *restrict i, int step, int y, int x)
{
  //return i + y * (DIM / TILE_H) + x;
  return NB_TILES_X * NB_TILES_Y * step + i + (y / TILE_H) * (DIM / TILE_H) + (x / TILE_W);
  //return DIM * DIM * step + i + (y / TILE_H) * DIM + x;
}

#define is_steady(step, y, x) (*is_steady_machin((step), (y), (x)))

bool* is_steady_machin(int step, int y, int x)
{
  
  return is_steady_tile(LA_TABLE, step, y, x);
}

void print_table()
{
  for (int y = 0; y < DIM; y+=TILE_H)
  {
      for (int x = 0; x < DIM; x+=TILE_W)
      {
        printf("%d ", is_steady(in, y, x));
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
        // if(j == (x + width - 1) && width != (TILE_W - 1))
        // {
        //   is_steady(in, y, x + TILE_W) = false;
        //   is_steady(out, y, x + TILE_W) = false;
        // }

        // if(j == x && width != (TILE_W - 1))
        // {
        //   is_steady(in, y, x - TILE_W) = false;
        //   is_steady(out, y, x - TILE_W) = false;
        // }

        // if(i == (y + height - 1) && height != (TILE_H - 1)) 
        // {
        //   is_steady(in, y + TILE_H, x) = false;
        //   is_steady(out, y + TILE_H, x) = false;
        // }

        // if(i == y && height != (TILE_H - 1)) 
        // {
        //   is_steady(in, y - TILE_H, x) = false;
        //   is_steady(out, y - TILE_H, x) = false;
        // }
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
bool check_steady(int step, int y, int x)
{
  if(x == 0 && y != 0 && y != DIM - TILE_H){
    return is_steady(step, y + TILE_H, x) && is_steady(step, y - TILE_H, x) && is_steady(step, y, x + TILE_W);
  }
  if(y == 0 && x != 0 && x != DIM - TILE_W){
    return is_steady(step, y + TILE_H, x) && is_steady(step, y, x - TILE_W) && is_steady(step, y, x + TILE_W);
  }
  if(x == DIM - TILE_W && y != 0 && y != DIM - TILE_H){
    return is_steady(step, y + TILE_H, x) && is_steady(step, y - TILE_H, x) && is_steady(step, y, x - TILE_W);
  }
  if(y == DIM - TILE_H && x != 0 && x != DIM - TILE_W){
    return is_steady(step, y - TILE_H, x) && is_steady(step, y, x + TILE_W) && is_steady(step, y, x - TILE_W);
  }

  if(y == 0 && x == 0){
    return is_steady(step, y + TILE_H, x) && is_steady(step, y, x + TILE_W);
  }
  if(y == 0 && x == DIM - TILE_W){
    return is_steady(step, y + TILE_H, x) && is_steady(step, y, x - TILE_W);
  }
  if(y == DIM - TILE_H && x == 0){
    return is_steady(step, y - TILE_H, x) && is_steady(step, y, x + TILE_W);
  }
  if(y == DIM - TILE_H && x == DIM - TILE_W){
    return is_steady(step, y - TILE_H, x) && is_steady(step, y, x - TILE_W);
  }

  return is_steady(step, y - TILE_H, x) && is_steady(step, y + TILE_H, x) && is_steady(step, y, x - TILE_W) && is_steady(step, y, x + TILE_W);
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
        if(is_steady(in, y, x) && is_steady(out, y, x)) continue;

        int diff = do_tile(x + (x == 0), y + (y == 0),
                    TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                    TILE_H - ((y + TILE_H == DIM) + (y == 0)));

        change |= diff;

        if(diff == 0 && (firstCheck || check_steady(in, y, x)))//
        {
          #pragma omp atomic
          is_steady(in, y, x) |= true;
        }
        else
        {
          if (!(x + TILE_W == DIM))
          {
            is_steady(in, y, x + TILE_W) = false;
            is_steady(out, y, x + TILE_W) = false;
          }

          if (!(x == 0))
          {
            is_steady(in, y, x - TILE_W) = false;
            is_steady(out, y, x - TILE_W) = false;
          }

          if (!(y + TILE_H == DIM))
          {
            is_steady(in, y + TILE_H, x) = false;
            is_steady(out, y + TILE_H, x) = false;
          }

          if (!(y == 0))
          {
            is_steady(in, y - TILE_H, x) = false;
            is_steady(out, y - TILE_H, x) = false;
          }
        }
      }
    // print_table();
    
    firstCheck = false;//


    swap_tables();
    if (change == 0)
    {
      // bool is_done = true;
      // for (int y = 0; y < DIM/TILE_H; y++)
      //   for (int x = 0; x < DIM/TILE_W; x++){
      //     if(!is_steady(in, y, x) || !is_steady(out, y, x)) 
      //     {
      //       is_done = false;
      //       break;
      //     }
      //   }
      // if(!is_done)
      // {
      //   printf("not done it=%d\n", it);
      //   continue;
      // }

      return it;
    }
  }

  return 0;
}



#pragma endregion
