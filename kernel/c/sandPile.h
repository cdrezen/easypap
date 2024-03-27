#ifndef SANDPILE
#define SANDPILE

#include "easypap.h"

#include <omp.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <unistd.h>


typedef unsigned int TYPE;

static TYPE* restrict TABLE = NULL;
//static TYPE* TABLE = NULL;

static inline TYPE *atable_cell(TYPE *restrict i, int y, int x);

#define atable(y, x) (*atable_cell(TABLE, (y), (x)))

static inline TYPE *table_cell(TYPE *restrict i, int step, int y, int x);

#define table(step, y, x) (*table_cell(TABLE, (step), (y), (x)))

static int in = 0;
static int out = 1;

static inline void swap_tables();

#define RGB(r, g, b) rgba(r, g, b, 0xFF)

static TYPE max_grains;

void asandPile_refresh_img();

/////////////////////////////  Initial Configurations

static inline void set_cell(int y, int x, unsigned v);

void asandPile_draw_4partout(void);

void asandPile_draw(char *param);

void ssandPile_draw(char *param);

void asandPile_draw_4partout(void);

void asandPile_draw_DIM(void);

void asandPile_draw_alea(void);

void asandPile_draw_big(void);

static void one_spiral(int x, int y, int step, int turns);

static void many_spirals(int xdebut, int xfin, int ydebut, int yfin, int step, int turns);

static void spiral(unsigned twists);

void asandPile_draw_spirals(void);

// shared functions

#define ALIAS(fun)       \
  void ssandPile_##fun() \
  {                      \
    asandPile_##fun();   \
  }

ALIAS(refresh_img);
ALIAS(draw_4partout);
ALIAS(draw_DIM);
ALIAS(draw_alea);
ALIAS(draw_big);
ALIAS(draw_spirals);

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Synchronous Kernel
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

void ssandPile_init();

void ssandPile_finalize();

int ssandPile_do_tile_default(int x, int y, int width, int height);

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned ssandPile_compute_seq(unsigned nb_iter);

unsigned ssandPile_compute_tiled(unsigned nb_iter);

#ifdef ENABLE_OPENCL

// Only called when --dump or --thumbnails is used
void ssandPile_refresh_img_ocl();

#endif

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Asynchronous Kernel
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

void asandPile_init();

void asandPile_finalize();

///////////////////////////// Version séquentielle simple (seq)
// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0

int asandPile_do_tile_default(int x, int y, int width, int height);

unsigned asandPile_compute_seq(unsigned nb_iter);

unsigned asandPile_compute_tiled(unsigned nb_iter);

#endif /* SANDPILE */