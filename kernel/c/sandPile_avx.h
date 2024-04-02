#include "sandPile.h"
#include <immintrin.h>

//#if __AVX2__ == 1

void ssandPile_tile_check_avx (void)
{
  // Tile width must be larger than AVX vector size
  easypap_vec_check (AVX_VEC_SIZE_FLOAT, DIR_HORIZONTAL);
}

//OMP_NUM_THREADS=1 ./run -k ssandPile -s 128 -v tmpavx -wt avx1 -n -du -ts 8 -i 4242
#pragma GCC optimize ("unroll-loops")
int ssandPile_do_tile_avx1(int x, int y, int width, int height)
{
  int diff = 0;
  const __m256i THREE_VEC = _mm256_set1_epi32(3);
  //const __m256i ZERO_VEC = _mm256_set1_epi32(0);

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j += AVX_VEC_SIZE_FLOAT) {

      __m256i cell_in, cell_out, cell_left, cell_right, cell_top, cell_bottom;

      cell_in = _mm256_loadu_si256 ((__m256i *)&table(in, i, j));
      cell_out = _mm256_loadu_si256 ((__m256i *)&table(out, i, j));

      cell_left = _mm256_loadu_si256 ((__m256i *)&table(in, i, j - 1));
      cell_right = _mm256_loadu_si256 ((__m256i *)&table(in, i, j + 1));
      cell_top = _mm256_loadu_si256 ((__m256i *)&table(in, i + 1, j));
      cell_bottom = _mm256_loadu_si256 ((__m256i *)&table(in, i - 1, j));

      //TODO: peut etre trouver une fonction du genre fmadd
      //TODO: utiliser des mm512

      //cell_out = cell_in % 4 (= &3):
      cell_out = _mm256_and_si256(cell_in, THREE_VEC);

      // neighbors /4 (= >>2):
      //TODO: voir si moins couteux avec factorisation + aroundi ou avec 2 mm512
      cell_left = _mm256_srli_epi32(cell_left, 2);
      cell_right = _mm256_srli_epi32(cell_right, 2);
      cell_top = _mm256_srli_epi32(cell_top, 2);
      cell_bottom = _mm256_srli_epi32(cell_bottom, 2);

      // cell_out += neighbors / 4
      cell_out = _mm256_add_epi32(cell_out, cell_left);
      cell_out = _mm256_add_epi32(cell_out, cell_right);
      cell_out = _mm256_add_epi32(cell_out, cell_top);
      cell_out = _mm256_add_epi32(cell_out, cell_bottom);

      _mm256_storeu_si256((__m256i *)&table(out, i, j), cell_out);

      ///diff = cell_in != cell_out :

      // //AVX
      // __m256 c_in = _mm256_castsi256_ps(cell_in);//(cast 0 cost)
      // __m256 c_out = _mm256_castsi256_ps(cell_out);
      // __m256 change = _mm256_cmp_ps(c_in, c_out, _CMP_NEQ_OS);//mask[8] (sz 256) : cell_in != cell_out
      // diff = _mm256_movemask_ps(change);//mask -> 8 bit

      //AVX512
      __mmask8 mask = _mm256_cmpneq_epu32_mask(cell_in, cell_out);// cell_in != cell_out : 0 modif -> 00000000 (avx512)
      diff = (int)mask;//__mmask8 = uint_8
      //diff = _popcnt32(mask);//compte les bits à 1 correspondant à une modif dans une des 8 cellules comparées. popcnt pas nécessaire au fonctionement ajd mais plus cohérent      
    }

  return  diff;
}

unsigned ssandPile_compute_tmpavx(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    int change = 0;
  #pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change)
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
      {

        if ((x + TILE_W == DIM))// || (y + TILE_H == DIM) || (y == 0) || (x == 0))
        {
          change |= ssandPile_do_tile_opt(x + (x == 0), y + (y == 0),
                                          TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                                          TILE_H - ((y + TILE_H == DIM) + (y == 0)));
        }
        else 
        {
          change |= do_tile(x + (x == 0), y + (y == 0),
                    TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                    TILE_H - ((y + TILE_H == DIM) + (y == 0)));
        }
      }

    swap_tables();
    if (change == 0) return it;
  }

  return 0;
}

int ssandPile_do_tile_avx(int x, int y, int width, int height)
{
    int diff = 0;

    for (int i = y; i < y + height; i++)
    {
        for (int j = x; j < x + width; j += 8)
        {
            __m256i in_values = _mm256_loadu_si256((__m256i *)&table(in, i - 1, j));
            __m256i next_in_values = _mm256_loadu_si256((__m256i *)&table(in, i + 1, j));
            __m256i left_in_values = _mm256_loadu_si256((__m256i *)&table(in, i, j - 1));
            __m256i right_in_values = _mm256_loadu_si256((__m256i *)&table(in, i, j + 1));

            __m256i out_ = _mm256_add_epi32(in_values, next_in_values);
            out_ = _mm256_add_epi32(out_, left_in_values);
            out_ = _mm256_add_epi32(out_, right_in_values);
            out_ = _mm256_and_si256(out_, _mm256_set1_epi32(3));

            __m256i current_values = _mm256_loadu_si256((__m256i *)&table(in, i, j));

            __m256i res = _mm256_cmpgt_epi32(out_, current_values);

            _mm256_storeu_si256((__m256i *)&table(out, i, j), out_);
            if (!_mm256_testz_si256(res, res))
                diff = 1;
        }
    }

    return diff;
}

unsigned asandPile_compute_tmpavx(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    int change = 0;
#pragma omp parallel for collapse(2) schedule(runtime) reduction(|:change)
    for (int y = 0; y < DIM; y += 2*TILE_H)
      for (int x = 0; x < DIM; x += 2*TILE_W)
      {
        if ((x + TILE_W == DIM) || (y + TILE_H == DIM) || (y == 0) || (x == 0))
        {
            //////////////////////////////
        }

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
      for (int x = TILE_W; x < DIM; x += 2*TILE_W)
      {

        if ((x + TILE_W == DIM) || (y + TILE_H == DIM) || (y == 0) || (x == 0))
        {
            //////////////////////////////
        }
        
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

//OMP_NUM_THREADS=1 ./run -k ssandPile -s 128 -v tmpavx -wt avx1 -n -du -ts 8 -i 4242
#pragma GCC optimize ("unroll-loops")
int asandPile_do_tile_avx1(int x, int y, int width, int height)
{
  int diff = 0;
  const __m256i THREE_VEC = _mm256_set1_epi32(3);
  const __m256i FOUR_VEC = _mm256_set1_epi32(4);
  //const __m256i ZERO_VEC = _mm256_set1_epi32(0);



  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j += AVX_VEC_SIZE_FLOAT) 
    {
      __m256i cell, cell_left, cell_right, cell_top, cell_bottom;


      cell = _mm256_loadu_si256 ((__m256i *)&atable(i, j));//Tj,i


      __mmask8 mask_leq = _mm256_cmple_epu32_mask(cell, THREE_VEC);//cell <= 3
      if(mask_leq) continue;


      diff = mask_leq;

      cell_left = _mm256_loadu_si256 ((__m256i *)&atable(i, j - 1));//Tj-1,i
      cell_right = _mm256_loadu_si256 ((__m256i *)&atable(i, j + 1));//Tj+1,i

      cell_top = _mm256_loadu_si256 ((__m256i *)&atable(i + 1, j));//Tj,i+k+1??
      cell_bottom = _mm256_loadu_si256 ((__m256i *)&atable(i - 1, j));//Tj,i-1

      // D <- Tij / 4
      __m256i D = _mm256_srli_epi32(cell, 2);

   
      //Tij = Tij % 4 + (D << 1) + (D >> 1):

      cell = _mm256_and_si256(cell, THREE_VEC);// Tij = Tij % 4
      cell = _mm256_add_epi32(cell, _mm256_slli_epi32(D, 1));//+= D << 1
      cell = _mm256_add_epi32(cell, _mm256_srli_epi32(D, 1));//+= D >> 1

    
      cell_left = _mm256_add_epi32(cell_left, D);//Tj-1,i += D
      cell_right = _mm256_add_epi32(cell_left, D);;//Tj+1,i += D


      cell_bottom = _mm256_add_epi32(cell_bottom, D);//Tj,i-1 += D[0]?
      cell_top = _mm256_add_epi32(cell_top, D);//Tj,i+k+1?? += D[k] k serait 7 (8bit) ?
      
      _mm256_storeu_si256((__m256i *)&atable(i - 1, j), cell_bottom);
      _mm256_storeu_si256((__m256i *)&atable(i + 1, j), cell_top);

      _mm256_storeu_si256((__m256i *)&atable(i, j - 1), cell_left);
      _mm256_storeu_si256((__m256i *)&atable(i, j), cell);
      _mm256_storeu_si256((__m256i *)&atable(i, j + 1), cell_right);      
    }

  return  diff;
}