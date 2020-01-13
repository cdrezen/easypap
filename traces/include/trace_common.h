#ifndef TRACE_COMMON_IS_DEF
#define TRACE_COMMON_IS_DEF

#define MAX_COLORS 13

extern unsigned cpu_colors[];

#define TRACE_BEGIN_ITER   0x101
#define TRACE_BEGIN_TILE   0x102
#define TRACE_END_TILE     0x103
#define TRACE_NB_CORES     0x104
#define TRACE_NB_ITER      0x105
#define TRACE_DIM          0x106
#define TRACE_END_ITER     0x107
#define TRACE_LABEL        0x108

#endif
