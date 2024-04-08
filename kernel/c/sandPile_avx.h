#include "sandPile.h"
#include <immintrin.h>

//#if __AVX2__ == 1

void ssandPile_tile_check_avx (void)
{
  // Tile width must be larger than AVX vector size
  easypap_vec_check (AVX512_VEC_SIZE_FLOAT, DIR_HORIZONTAL);
}

void asandPile_tile_check_avx (void)
{
  // Tile width must be larger than AVX vector size
  easypap_vec_check (AVX512_VEC_SIZE_FLOAT, DIR_HORIZONTAL);
}

void ssandPile_tile_check_avx_256 (void)
{
  // Tile width must be larger than AVX vector size
  easypap_vec_check (AVX_VEC_SIZE_FLOAT, DIR_HORIZONTAL);
}

void asandPile_tile_check_avx_256 (void)
{
  // Tile width must be larger than AVX vector size
  easypap_vec_check (AVX_VEC_SIZE_FLOAT, DIR_HORIZONTAL);
}

//OMP_NUM_THREADS=32 OMP_SCHEDULE=static ./run -k ssandPile -s 8192 -v lazy -wt avx -ts 16 -a alea -n -du -sh
int ssandPile_do_tile_avx(int x, int y, int width, int height)
{
    unsigned diff = 0;
    const __m512i THREE_VEC = _mm512_set1_epi32(3);
    const __m512i LAST_VEC_J = _mm512_set1_epi32(DIM - AVX512_VEC_SIZE_FLOAT);

    for (int i = y; i < y + height; i++)
    {
        for (int j = x; j < x + width; j += AVX512_VEC_SIZE_FLOAT) // Utilisation de pas de 16 pour AVX-512
        {
            __m512i cell_in = _mm512_loadu_si512((__m512i *)&table(in, i, j));
            __m512i cell_out = _mm512_loadu_si512((__m512i *)&table(out, i, j));

            __m512i cell_left = _mm512_loadu_si512((__m512i *)&table(in, i, j - 1));
            __m512i cell_right = _mm512_loadu_si512((__m512i *)&table(in, i, j + 1));
            __m512i cell_top = _mm512_loadu_si512((__m512i *)&table(in, i + 1, j));
            __m512i cell_bottom = _mm512_loadu_si512((__m512i *)&table(in, i - 1, j));

            // Calcul de cell_out = cell_in % 4 (= &3) + neighbors / 4
            cell_out = _mm512_and_epi32(cell_in, THREE_VEC);
            cell_out = _mm512_add_epi32(cell_out, _mm512_srli_epi32(cell_top, 2));
            cell_out = _mm512_add_epi32(cell_out, _mm512_srli_epi32(cell_bottom, 2));
            cell_out = _mm512_add_epi32(cell_out, _mm512_srli_epi32(cell_left, 2));

            const __m512i J_VEC = _mm512_set1_epi32(j);
            __mmask16 mask_last = _mm512_cmpeq_epu32_mask(J_VEC, LAST_VEC_J);// = if(j != DIM - AVX512_VEC_SIZE_FLOAT) :

            __m512i regular = _mm512_add_epi32(cell_out, _mm512_srli_epi32(cell_right, 2));//if
            __m512i last_tile_case = _mm512_mask_add_epi32(cell_out, 0x3FFF, cell_out, _mm512_srli_epi32(cell_right, 2));//else

            cell_out = _mm512_mask_blend_epi32(mask_last, regular, last_tile_case);
             //cell_out = _mm512_mask_add_epi32(cell_out, _mm512_kand(0x3FFF, mask_last), cell_out, _mm512_srli_epi32(cell_right, 2));

            // Stockage du résultat
            _mm512_storeu_si512((__m512i *)&table(out, i, j), cell_out);

            // Comparaison de cell_in et cell_out pour détecter les modifications
            __mmask16 mask = _mm512_cmpneq_epu32_mask(cell_in, cell_out);
            diff |= mask;
        }
    }

    //printf("j=%d width=%d x+width=%d DIM-AVX512_VEC_SIZE_FLOAT=%d\n", j, width, x+width, DIM - AVX512_VEC_SIZE_FLOAT);

    return diff;
}

//OMP_NUM_THREADS=1 ./run -k ssandPile -s 128 -v tmpavx -wt avx1 -n -du -ts 8 -i 4242
//OMP_NUM_THREADS=43 OMP_SCHEDULE=static ./run -k ssandPile -s 1024 -i 10000 -v tmpavx -wt avx_256 -th 8 -tw 1024 -n -ft
int ssandPile_do_tile_avx_256(int x, int y, int width, int height)
{
  unsigned diff = 0;
  const __m256i THREE_VEC = _mm256_set1_epi32(3);
  const __m256i LAST_VEC_J = _mm256_set1_epi32(DIM - AVX_VEC_SIZE_FLOAT);


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

      // cell_out += neighbors / 4 :
      
      cell_out = _mm256_add_epi32(cell_out, cell_left);
      cell_out = _mm256_add_epi32(cell_out, cell_top);
      cell_out = _mm256_add_epi32(cell_out, cell_bottom);

      const __m256i J_VEC = _mm256_set1_epi32(j);
      __mmask8 mask_last = _mm256_cmpeq_epu32_mask(J_VEC, LAST_VEC_J);// = if(j != DIM - AVX512_VEC_SIZE_FLOAT) :

      __m256i regular = _mm256_add_epi32(cell_out, cell_right);//if
      __m256i last_tile_case = _mm256_mask_add_epi32(cell_out, 0x3F, cell_out, cell_right);//else

      cell_out = _mm256_mask_blend_epi32(mask_last, regular, last_tile_case);

      _mm256_storeu_si256((__m256i *)&table(out, i, j), cell_out);

      ///diff = cell_in != cell_out :

      //AVX512
      __mmask8 mask = _mm256_cmpneq_epu32_mask(cell_in, cell_out);// cell_in != cell_out : 0 modif -> 00000000 (avx512)
      diff |= mask;//__mmask8 = uint_8
      //diff = _popcnt32(mask);//compte les bits à 1 correspondant à une modif dans une des 8 cellules comparées. popcnt pas nécessaire au fonctionement ajd mais plus cohérent      
    }

  return  diff;
}

//OMP_NUM_THREADS=1 OMP_SCHEDULE=static ./run -k ssandPile -s 512 -v lazy1 -wt avx_spirals -n -sh -a spirals -ts 16
int ssandPile_do_tile_avx_spirals(int x, int y, int width, int height)
{
    int diff = 0;
    const __m512i THREE_VEC = _mm512_set1_epi32(3);
    const __m512i ZERO_VEC = _mm512_set1_epi8(0);
    const __m512i LAST_VEC_J = _mm512_set1_epi32(DIM - AVX512_VEC_SIZE_FLOAT);

    int j = x;

    for (int i = y; i < y + height; i++)
    {
        for (j = x; j < x + width; j += AVX512_VEC_SIZE_FLOAT) // Utilisation de pas de 16 pour AVX-512
        {
            __m512i cell_in = _mm512_loadu_si512((__m512i *)&table(in, i, j));
            __m512i cell_out = _mm512_loadu_si512((__m512i *)&table(out, i, j));

            __m512i cell_left = _mm512_loadu_si512((__m512i *)&table(in, i, j - 1));
            __m512i cell_right = _mm512_loadu_si512((__m512i *)&table(in, i, j + 1));
            __m512i cell_top = _mm512_loadu_si512((__m512i *)&table(in, i + 1, j));
            __m512i cell_bottom = _mm512_loadu_si512((__m512i *)&table(in, i - 1, j));

            // Calcul de cell_out = cell_in % 4 (= &3) + neighbors / 4
            cell_out = _mm512_and_epi32(cell_in, THREE_VEC);
            cell_out = _mm512_add_epi32(cell_out, _mm512_srli_epi32(cell_top, 2));
            cell_out = _mm512_add_epi32(cell_out, _mm512_srli_epi32(cell_bottom, 2));
            cell_out = _mm512_add_epi32(cell_out, _mm512_srli_epi32(cell_left, 2));

            const __m512i J_VEC = _mm512_set1_epi32(j);
            __mmask16 mask_last = _mm512_cmpeq_epu32_mask(J_VEC, LAST_VEC_J);// = if(j != DIM - AVX512_VEC_SIZE_FLOAT) :

            __m512i regular = _mm512_add_epi32(cell_out, _mm512_srli_epi32(cell_right, 2));//if
            __m512i last_tile_case = _mm512_mask_add_epi32(cell_out, 0x3FFF, cell_out, _mm512_srli_epi32(cell_right, 2));//else

            cell_out = _mm512_mask_blend_epi32(mask_last, regular, last_tile_case);
            

            // Stockage du résultat
            _mm512_storeu_si512((__m512i *)&table(out, i, j), cell_out);

            // Comparaison de cell_in et cell_out pour détecter les modifications
            __mmask16 mask = _mm512_cmpneq_epu32_mask(cell_in, cell_out);
            if(mask != 0)//TODO:!ktestz
            {
              
              is_steady(out, y, x + AVX512_VEC_SIZE_FLOAT) &= false;
              //_mm512_storeu_epi8((__m512i *)&is_steady(out, y, x), ZERO_VEC);
              
              if (j == x && x != 1){
                is_steady(out, y, x - TILE_W) &= false;
              }

              if (j == (x + width - AVX512_VEC_SIZE_FLOAT)){// && (x + width - 1) != DIM - 1) {
                is_steady(out, y, x + TILE_W) &= false;
              }

              if (i == y && y != 1){
                is_steady(out, y - TILE_H, x) &= false;
              }
        
              if (i == (y + height-1)){// && ((y + height - 1) != DIM - 1)){
                is_steady(out, y + TILE_H, x) &= false;;
              }
              
              diff = 1;
            }

        }
    }

    return diff;
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

unsigned asandPile_compute_tmpavx(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    int change = 0;

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
      {
        if ((x + TILE_W == DIM) || (y + TILE_H == DIM) || (y == 0) || (x == 0))
        {
        change |=
            asandPile_do_tile_opt1(x + (x == 0), y + (y == 0),
                    TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                    TILE_H - ((y + TILE_H == DIM) + (y == 0)));
        }
        else
        {
          change |=
            do_tile(x + (x == 0), y + (y == 0),
                    TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                    TILE_H - ((y + TILE_H == DIM) + (y == 0)));
        }
      }
    if (change == 0)
      return it;
  }

  return 0;
}

//OMP_NUM_THREADS=1 OMP_SCHEDULE=static ./run -k asandPile -s 128 -v tmpavx -wt avx -ts 16 -n
//OMP_NUM_THREADS=1 OMP_SCHEDULE=static ./run -k asandPile -s 128 -v tiled -wt avx -ts 16 -n
//OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./run -k asandPile -s 512 -v omp -wt avx -ts 16
//OMP_NUM_THREADS=16 OMP_SCHEDULE=static ./run -k asandPile -s 512 -v omp -wt avx -ts 32
//bug synchro
int asandPile_do_tile_avx(int x, int y, int width, int height)
{
  if((x == AVX512_VEC_SIZE_FLOAT) || (x == DIM - AVX512_VEC_SIZE_FLOAT)
  || (y == AVX512_VEC_SIZE_FLOAT) || (y == DIM - AVX512_VEC_SIZE_FLOAT)) {
    return asandPile_do_tile_opt1(x, y, width, height);
  }

  unsigned diff = 0;
  const __m512i THREE_VEC = _mm512_set1_epi32(3);
  const __m512i FOUR_VEC = _mm512_set1_epi32(4);
  const __m512i ZERO_VEC = _mm512_set1_epi32(0);

  TYPE* ptr = atable_cell(TABLE, y - 1, x);

  #pragma omp simd collapse(2) reduction(|:diff) safelen(AVX512_VEC_SIZE_FLOAT*5) aligned(ptr:32*5)// private(Dk)
  for (int i = y; i < y + height; i++){
    for (int j = x; j < x + width; j += AVX512_VEC_SIZE_FLOAT) 
    {

      __m512i cells = _mm512_loadu_si512((__m512i *)&atable(i, j));
      __m512i last_cells = cells;

      // __mmask16 mask_leq = _mm512_cmpge_epu32_mask(THREE_VEC, cells);//cell >= 4
      // if (mask_leq == 1) continue;
    
      __m512i cells_top = _mm512_loadu_si512 ((__m256i *)&atable(i + 1, j));    
      __m512i cells_bottom = _mm512_loadu_si512((__m256i *)&atable(i - 1, j)); 

     

      __m512i D = _mm512_srli_epi32(cells, 2);

      // %4 + D<<1 +D>>1 :
      cells = _mm512_and_epi32(cells, THREE_VEC);
      // >> 1
      cells = _mm512_add_epi32(cells, _mm512_mask_alignr_epi32(ZERO_VEC, 0x7FFF, ZERO_VEC, D, 1));//_mm512_alignr_epi32(ZERO_VEC, D, 1));//plus rapide avec mask ici
      // << 1
      cells = _mm512_add_epi32(cells, _mm512_alignr_epi32(D, ZERO_VEC, AVX512_VEC_SIZE_FLOAT - 1));

      cells_top = _mm512_add_epi32(cells_top, D);
      cells_bottom = _mm512_add_epi32(cells_bottom, D);

      //__m512i cells_left = _mm512_loadu_si512((__m512i *)&atable(i, j - 1));
      //__m512i cells_right = _mm512_loadu_si512((__m512i *)&atable(i, j + 1));


      TYPE Dk[16] = { 0 }; 
      _mm512_storeu_epi32(&Dk, D);

      //Tj-1,i += D[0]:
      //#pragma omp atomic
      atable(i, j - 1) += Dk[0];//_mm512_cvtsi512_si32(D);
      //__m512i Dk0_VEC = _mm512_set1_epi32(Dk[0]);
      //cells_left = _mm512_mask_add_epi32(cells_left, 0x0001, cells_left, D);
      //_mm512_storeu_si512((__m512i *)&atable(i, j - 1), cells_left);

      //Tj+1+k,i += D[k] :
      //#pragma omp atomic
      atable(i, j + AVX512_VEC_SIZE_FLOAT) += Dk[AVX512_VEC_SIZE_FLOAT - 1];//if(x == DIM - AVX512_VEC_SIZE_FLOAT)?
      //__m512i Dkk_VEC = _mm512_set1_epi32(Dk[AVX512_VEC_SIZE_FLOAT - 1]);
      //cells_right = _mm512_mask_add_epi32(cells_right, 0x8000, cells_right, D);
      //_mm512_storeu_si512((__m512i *)&atable(i, j + 1), cells_right);

      _mm512_storeu_si512((__m512i *)&atable(i - 1, j), cells_bottom);
      _mm512_storeu_si512((__m512i *)&atable(i, j), cells);
      _mm512_storeu_si512((__m512i *)&atable(i + 1, j), cells_top);

      __mmask16 mask = _mm512_cmpneq_epu32_mask(cells, last_cells);
      //#pragma omp atomic
      diff |= mask;
    }
  }

  return diff;
}



//OMP_NUM_THREADS=1 ./run -k asandPile -s 128 -v tiled -wt avx_256 -n -du -ts 8
#pragma GCC optimize ("unroll-loops")
int asandPile_do_tile_avx_256(int x, int y, int width, int height)
{
  if((x == AVX512_VEC_SIZE_FLOAT) || (x == DIM - AVX512_VEC_SIZE_FLOAT)
  || (y == AVX512_VEC_SIZE_FLOAT) || (y == DIM - AVX512_VEC_SIZE_FLOAT)) {
    return asandPile_do_tile_opt1(x, y, width, height);
  }

  unsigned diff = 0;
  const __m256i THREE_VEC = _mm256_set1_epi32(3);
  const __m256i FOUR_VEC = _mm256_set1_epi32(4);
  const __m256i ZERO_VEC = _mm256_set1_epi32(0);

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j += AVX_VEC_SIZE_FLOAT) 
    {
      __m256i cell, last_cell, cell_top, cell_bottom;


      cell = _mm256_loadu_si256 ((__m256i *)&atable(i, j));//Tj,i
      last_cell = cell;

      // __mmask16 mask_leq = _mm256_cmpge_epu32_mask(THREE_VEC, cell);//cell >= 4
      // if (mask_leq == 1) continue;

      cell_bottom = _mm256_loadu_si256((__m256i *)&atable(i - 1, j)); // Tj-1,i
      cell_top = _mm256_loadu_si256((__m256i *)&atable(i + 1, j));    // Tj+1,i

      // D <- Tij / 4
      __m256i D = _mm256_srli_epi32(cell, 2);

      cell = _mm256_and_si256(cell, THREE_VEC);// Tij = Tij % 4
      cell = _mm256_add_epi32(cell, _mm256_mask_alignr_epi32(ZERO_VEC, 0x7F, ZERO_VEC, D, 1));//+= D >> 1
      cell = _mm256_add_epi32(cell, _mm256_mask_alignr_epi32(ZERO_VEC, 0xFE, D, ZERO_VEC, AVX_VEC_SIZE_FLOAT - 1));//+= D << 1

      cell_bottom = _mm256_add_epi32(cell_bottom, D); // Tj-1,i += D
      cell_top = _mm256_add_epi32(cell_top, D);// Tj+1,i += D

      atable(i, j - 1) += _mm256_extract_epi32(D, 0);// Tj-1,i += D
      //if(j != DIM - AVX_VEC_SIZE_FLOAT)?
      atable(i, j + AVX_VEC_SIZE_FLOAT) += _mm256_extract_epi32(D, AVX_VEC_SIZE_FLOAT-1);// Tj+1,i += D

      _mm256_storeu_si256((__m256i *)&atable(i - 1, j), cell_bottom);
      _mm256_storeu_si256((__m256i *)&atable(i, j), cell);
      _mm256_storeu_si256((__m256i *)&atable(i + 1, j), cell_top);

      __mmask16 mask = _mm256_cmpneq_epu32_mask(cell, last_cell);
      diff = mask;
    }

  return  diff;
}




//(vieux code)
// cells_left = _mm512_mask_add_epi32(cells_left, 0x0001, cells_left, D);
// _mm512_mask_storeu_epi32((__m512i *)&atable(i, j - 1), 0x0001, cells_left);
// cells_right = _mm512_add_epi32(cells_right, D);
// _mm512_mask_storeu_epi32((__m512i *)&atable(i, j + 1), 0x8000, cells_right);

// cells_left = _mm512_add_epi32(cells_left, D);
// cells_right = _mm512_add_epi32(cells_right, D);
// _mm512_storeu_si512((__m512i *)&atable(i, j - 1), cells_left);
// _mm512_storeu_si512((__m512i *)&atable(i, j + 1), cells_right);
      