#include "sandPile.h"

#pragma region 4.4 // Lazy OpenMP implementations ssand

static bool* restrict LA_TABLE = NULL;//tableau de booleen pour le LAzy

static inline bool* is_steady_tile(bool *restrict i, int step, int y, int x)
{
  return NB_TILES_X * NB_TILES_Y * step + i + (y / TILE_H) * (DIM / TILE_H) + (x / TILE_W);
}

#define is_steady(step, y, x) (*is_steady_tile(LA_TABLE, (step), (y), (x)))

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
  memset(LA_TABLE, 0, 2 * NB_TILES_X * NB_TILES_Y  * sizeof(bool));//met tout à false(0)
  // print_table();
}

void ssandPile_finish_lazy()
{
  ssandPile_finalize();

  free(LA_TABLE);
}


#pragma GCC optimize ("unroll-loops")
int ssandPile_do_tile_lazy(int x, int y, int width, int height)
{
  int diff = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
    {
      TYPE* cell_in = table_cell(TABLE, in, i, j);
      TYPE* cell_out = table_cell(TABLE, out, i, j);

      *cell_out = table(in, i, j) % 4
                 + (table(in, i, j - 1) / 4) 
                 + (table(in, i, j + 1) / 4) 
                 + (*(cell_in - DIM) / 4)  // table(in, i - 1, j) / 4
                 + (*(cell_in + DIM) / 4); // table(in, i + 1, j) / 4
      
      if (*cell_in != *cell_out)//(*cell_out >= 4) modifié pour tuile stagnate
      {
        diff = 1;
      }
    }

  return diff;
}

/* check_steady retourne true si la tuile (x,y) possede des voisins stables 
*  rappel : is_steady retourne true si la tuile est stable
*/
bool has_all_neighbors_steady(int step, int y, int x)
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

//retourne true si au moins une tuile voisine est active, false si tout les voisin sont stable.
bool has_active_neighbors(int step, int y, int x)
{
  return (!(x == 0) && !is_steady(step, y, x - TILE_W))
      || (!(x + TILE_W == DIM) && !is_steady(step, y, x + TILE_W))
      || (!(y + TILE_H == DIM) && !is_steady(step, y + TILE_H, x))
      || (!(y == 0) && !is_steady(step, y - TILE_H, x));

  // if (!(x == 0) && !is_steady(step, y, x - TILE_W)) return true;
  // if (!(x + TILE_W == DIM) && !is_steady(step, y, x + TILE_W)) return true;
  // if (!(y + TILE_H == DIM) && !is_steady(step, y + TILE_H, x)) return true;
  // if (!(y == 0) && !is_steady(step, y - TILE_H, x)) return true;
  // return false;
}

//OMP_NUM_THREADS=16 ./run -k ssandPile -s 512 -v lazy -wt lazy -n -sh -a alea -ts 32
unsigned ssandPile_compute_lazy(unsigned nb_iter)
{
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

        if(diff == 0)// && (firstCheck || has_active_neighbors(in, y, x)))//
        {
          #pragma omp atomic
          is_steady(in, y, x) |= true;
        }
        else
        {
          is_steady(out, y, x) &= false;

          if (!(x == 0))
            is_steady(out, y, x - TILE_W) &= false;

          if (!(x + TILE_W == DIM))
            is_steady(out, y, x + TILE_W) &= false;

          if (!(y + TILE_H == DIM))
            is_steady(out, y + TILE_H, x) &= false;

          if (!(y == 0))
            is_steady(out, y - TILE_H, x) &= false;
        }
      }

    swap_tables();
    if (change == 0) return it;
  }

  return 0;
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Lazy1 - Réactivation plus précise, meileurs perf spirals, pire alea
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

//TODO?: tableau d'objets à 4 ou 5 valeurs (tuile + bords) pour gerer l'inactivité des 4 bords plutot que tableau de 1bool/1tuile ?

void ssandPile_init_lazy1()
{
  ssandPile_init_lazy();
}

void ssandPile_finish_lazy1()
{
  ssandPile_finish_lazy();
}


static inline void ssandPile_activate_neighbor_lazy(int ij, int border_pos, int border_image, int y, int x)
{
    if(ij == border_pos && border_pos != border_image)
    {
      //#pragma omp atomic
      is_steady(out, y, x) &= false;
    }
}

static inline void ssandPile_activate_neighbor_lazy_ptr(int ij, int border_pos, int border_image, bool* ptr)
{
    if(ij == border_pos && border_pos != border_image)
    {
      //#pragma omp atomic
      *ptr &= false;
    }
}

#pragma GCC optimize ("unroll-loops")
int ssandPile_do_tile_lazy1(int x, int y, int width, int height)
{
  int diff = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
    {
      TYPE* cell_in = table_cell(TABLE, in, i, j);
      TYPE* cell_out = table_cell(TABLE, out, i, j);

      *cell_out = table(in, i, j) % 4
                 + (table(in, i, j - 1) / 4) 
                 + (table(in, i, j + 1) / 4) 
                 + (*(cell_in - DIM) / 4)  // table(in, i - 1, j) / 4
                 + (*(cell_in + DIM) / 4); // table(in, i + 1, j) / 4
      
      if (*cell_in != *cell_out)//(*cell_out >= 4) modifié pour tuile stagnate
      {
        diff = 1;

        is_steady(out, y, x) &= false;

        if (j == x && x != 1)
          is_steady(out, y, x - TILE_W) &= false;

        if (j == (x + width - 1) && ((x + width - 1) != DIM - 1))//(x != DIM - TILE_W) 
          is_steady(out, y, x + TILE_W) &= false;

        if (i == y && y != 1)
          is_steady(out, y - TILE_H, x) &= false;
          //*(is_steady_ptr-NB_TILES_Y) &= false;
        
        if (i == (y + height - 1) && ((y + height - 1) != DIM - 1))//(y != DIM - TILE_H) 
          is_steady(out, y + TILE_H, x) &= false;
          //*(is_steady_ptr+NB_TILES_Y) &= false;
          
        // ssandPile_activate_neighbor_lazy(j, x, 1, y, x - TILE_W);
        // ssandPile_activate_neighbor_lazy(j, x + width - 1, DIM - 1, y, x + TILE_W);
        // ssandPile_activate_neighbor_lazy(i, y, 1, y - TILE_H, x);
        // ssandPile_activate_neighbor_lazy(i, y + height - 1, DIM - 1, y + TILE_H, x);

        // bool* is_steady_ptr = is_steady_tile(LA_TABLE, out, y, x);
        // ssandPile_activate_neighbor_lazy_ptr(j, x, 1, is_steady_ptr-1);
        // ssandPile_activate_neighbor_lazy_ptr(j, x + width - 1, DIM - 1, is_steady_ptr+1);
        // ssandPile_activate_neighbor_lazy_ptr(i, y, 1, is_steady_ptr-NB_TILES_Y);
        // ssandPile_activate_neighbor_lazy_ptr(i, y + height - 1, DIM - 1, is_steady_ptr+NB_TILES_Y);
      }
    }

  return diff;
}

//OMP_NUM_THREADS=32 ./run -k ssandPile -s 512 -v lazy -wt lazy -n -sh -a alea -ts 32
unsigned ssandPile_compute_lazy1(unsigned nb_iter)
{
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

        if(diff == 0)
        {
          #pragma omp atomic
          is_steady(in, y, x) |= true;
        }
      }


    swap_tables();
    if (change == 0)
    {
      return it;
    }
  }

  return 0;
}


#pragma endregion
