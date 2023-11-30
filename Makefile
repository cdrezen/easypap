
PROGRAM	:= bin/easypap

# Must be the first rule
.PHONY: default
default: $(PROGRAM)

########## Config Section ##########

ENABLE_SDL			= 1
ENABLE_MONITORING	= 1
ENABLE_VECTO		= 1
ENABLE_TRACE		= 1
ENABLE_MPI			= 1
ENABLE_SHA			= 1
ENABLE_OPENCL		= 1
#ENABLE_MIPP			= 1
#ENABLE_CUDA			= 1
#ENABLE_PAPI			= 1

###### Customization Section #######

# Compilers
CC				:= gcc
CXX				:= g++

# Optimization level
CFLAGS 			:= -O3 -march=native

# Warnings
CFLAGS			+= -Wall -Wno-unused-function

####################################

CFLAGS			+= -I./include -I./traces/include
LDLIBS			+= -lm

CXXFLAGS			:= -std=c++11

OS_NAME			:= $(shell uname -s | tr a-z A-Z)
ARCH			:= $(shell uname -m | tr a-z A-Z)

SOURCES			:= $(wildcard src/*.c)

ifneq ($(ENABLE_SDL), 1)
SOURCES			:= $(filter-out src/gmonitor.c src/graphics.c src/cpustat.c, $(SOURCES))
endif

ifneq ($(ENABLE_PAPI), 1)
SOURCES			:= $(filter-out src/perfcounter.c, $(SOURCES))
endif

ifneq ($(ENABLE_OPENCL), 1)
SOURCES			:= $(filter-out src/ocl.c, $(SOURCES))
endif

ifneq ($(ENABLE_SHA), 1)
SOURCES			:= $(filter-out src/hash.c, $(SOURCES))
endif

CUDA_SOURCE	:= $(wildcard src/*.cu)

KERNELS			:= $(wildcard kernel/c/*.c)
CXX_KERNELS	 	:= $(wildcard kernel/mipp/*.cpp)
CUDA_KERNELS	:= $(wildcard kernel/cuda/*.cu)

T_SOURCE		:= traces/src/trace_common.c

ifeq ($(ENABLE_TRACE), 1)
T_SOURCE		+= traces/src/trace_record.c
endif

L_SOURCE		:= $(wildcard src/*.l)
L_GEN			:= $(L_SOURCE:src/%.l=obj/%.c)

OBJECTS			:= $(SOURCES:src/%.c=obj/%.o)
K_OBJECTS		:= $(KERNELS:kernel/c/%.c=obj/%.o)
CXX_K_OBJECTS	:= $(CXX_KERNELS:kernel/mipp/%.cpp=obj/mipp_%.o)
T_OBJECTS		:= $(T_SOURCE:traces/src/%.c=obj/%.o)
L_OBJECTS		:= $(L_SOURCE:src/%.l=obj/%.o)
CUDA_OBJECTS	:= $(CUDA_SOURCE:src/%.cu=obj/%.o)
CUDA_K_OBJECTS	:= $(CUDA_KERNELS:kernel/cuda/%.cu=obj/cuda_%.o)

ALL_OBJECTS	:= $(OBJECTS) $(K_OBJECTS) $(T_OBJECTS) $(L_OBJECTS)

ifeq ($(ENABLE_CUDA), 1)
ALL_OBJECTS		+= $(CUDA_OBJECTS) $(CUDA_K_OBJECTS)
NEED_CPP_LINK	:= 1
endif

ifeq ($(ENABLE_MIPP), 1)
ALL_OBJECTS		+= $(CXX_K_OBJECTS)
NEED_CPP_LINK	:= 1
endif

DEPENDS			:= $(SOURCES:src/%.c=deps/%.d)
K_DEPENDS		:= $(KERNELS:kernel/c/%.c=deps/%.d)
CXX_K_DEPENDS	:= $(CXX_KERNELS:kernel/mipp/%.cpp=deps/mipp_%.d)
T_DEPENDS		:= $(T_SOURCE:traces/src/%.c=deps/%.d)
L_DEPENDS		:= $(L_GEN:obj/%.c=deps/%.d)
CUDA_DEPENDS	:= $(CUDA_SOURCE:src/%.cu=deps/%.d)
CUDA_K_DEPENDS	:= $(CUDA_KERNELS:kernel/cuda/%.cu=deps/cuda_%.d)

ALL_DEPENDS		:= $(DEPENDS) $(K_DEPENDS) $(T_DEPENDS) $(L_DEPENDS)

ifeq ($(ENABLE_CUDA), 1)
ALL_DEPENDS		+= $(CUDA_DEPENDS) $(CUDA_K_DEPENDS)
endif

ifeq ($(ENABLE_MIPP), 1)
ALL_DEPENDS		+= $(CXX_K_DEPENDS)
endif

MAKEFILES		:= Makefile

ifeq ($(OS_NAME), DARWIN)
LDFLAGS			+= -Wl,-ld_classic
LDLIBS			+= -framework OpenGL
else
CFLAGS			+= -pthread -rdynamic
LDFLAGS			+= -export-dynamic
LDLIBS			+= -lGL -ldl
endif

# Vectorization
ifeq ($(ENABLE_VECTO), 1)
CFLAGS			+= -DENABLE_VECTO
endif

# Monitoring
ifeq ($(ENABLE_MONITORING), 1)
CFLAGS			+= -DENABLE_MONITORING
endif

# OpenMP
CFLAGS			+= -fopenmp
LDFLAGS			+= -fopenmp

# OpenCL
ifeq ($(ENABLE_OPENCL), 1)
CFLAGS			+= -DENABLE_OPENCL -DCL_SILENCE_DEPRECATION -DARCH=$(ARCH)_ARCH
ifeq ($(OS_NAME), DARWIN)
LDLIBS			+= -framework OpenCL
else
LDLIBS			+= -lOpenCL
endif
endif

# Hardware Locality (hwloc)
PACKAGES		:= hwloc

# Simple DirectMedia Layer (SDL)
ifeq ($(ENABLE_SDL), 1)
CFLAGS			+= -DENABLE_SDL
PACKAGES		+= SDL2_image SDL2_ttf
endif

ifeq ($(ENABLE_TRACE), 1)
# Right now, only fxt is supported
CFLAGS			+= -DENABLE_TRACE -DENABLE_FUT
PACKAGES		+= fxt
endif

# Message Passing Interface (MPI)
ifeq ($(ENABLE_MPI), 1)
CFLAGS			+= -DENABLE_MPI
PACKAGES		+= ompi
endif

# Performance Application Programming Interface (PAPI)
ifeq ($(ENABLE_PAPI), 1)
CFLAGS			+= -DENABLE_PAPI
MICROARCH		:= $(shell echo $(shell (gcc -march=native -Q --help=target) | grep -m 1 march) | cut -d ' ' -f2)
ifeq ($(MICROARCH), $(filter $(MICROARCH),skylake skylake-avx512 cascadelake))
CFLAGS 			+= -DMICROARCH_SKYLAKE
else
ifeq ($(MICROARCH), haswell)
CFLAGS 			+= -DMICROARCH_HASWELL
endif
endif
PACKAGES		+= papi
endif

# Secure Hash Algorithm (SHA)
ifeq ($(ENABLE_SHA), 1)
CFLAGS			+= -DENABLE_SHA
PACKAGES		+= openssl
endif

# Compute Unified Device Architecture (CUDA)
ifeq ($(ENABLE_CUDA), 1)
CFLAGS			+= -DENABLE_CUDA
LDLIBS			+= -lcudart
PACKAGES		+= cuda
CUDA_CFLAGS 	:= -O3 -I./include -I./traces/include
endif

# MyIntrinsics++ (MIPP)
ifeq ($(ENABLE_MIPP), 1)
CXXFLAGS		+= -I./lib/mipp/src
endif

# Query CFLAGS and LDLIBS for all packages
PKG_CHECK		:= $(shell if pkg-config --print-errors --exists $(PACKAGES); then echo 0; else echo 1; fi)

# If some packages are missing, abort make process
ifeq ($(PKG_CHECK), 1)
$(error Installation problem: missing package)
endif

CFLAGS			+= $(shell pkg-config --cflags $(PACKAGES))
LDFLAGS			+= $(shell pkg-config --libs-only-L $(PACKAGES))
LDLIBS			+= $(shell pkg-config --libs-only-l $(PACKAGES))

CXXFLAGS		+= $(CFLAGS)

ifeq ($(NEED_CPP_LINK), 1)
LD				:= $(CXX)
else
LD				:= $(CC)
endif


########## Compute rules ###########

$(ALL_OBJECTS): $(MAKEFILES)

$(PROGRAM): $(ALL_OBJECTS)
	$(LD) -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(OBJECTS): obj/%.o: src/%.c
	$(CC) -o $@ $(CFLAGS) -c $<

$(K_OBJECTS): obj/%.o: kernel/c/%.c
	$(CC) -o $@ $(CFLAGS) -c $<

ifeq ($(ENABLE_MIPP), 1)
$(CXX_K_OBJECTS): obj/mipp_%.o: kernel/mipp/%.cpp
	$(CXX) -o $@ $(CXXFLAGS) -c $<
endif

$(T_OBJECTS): obj/%.o: traces/src/%.c
	$(CC) -o $@ $(CFLAGS) -c $<

$(L_OBJECTS): obj/%.o: obj/%.c
	$(CC) -o $@ $(CFLAGS) -c $<

$(L_GEN): obj/%.c: src/%.l
	$(LEX) -t $< > $@

ifeq ($(ENABLE_CUDA), 1)
$(CUDA_OBJECTS): obj/%.o: src/%.cu
	nvcc -o $@ $(CUDA_CFLAGS) -c $<

$(CUDA_K_OBJECTS): obj/cuda_%.o: kernel/cuda/%.cu
	nvcc -o $@ $(CUDA_CFLAGS) -c $<
endif

.PHONY: depend
depend: $(ALL_DEPENDS)

$(ALL_DEPENDS): $(MAKEFILES)

$(DEPENDS): deps/%.d: src/%.c
	$(CC) $(CFLAGS) -MM -MT "deps/$*.d obj/$*.o" $< > $@

$(K_DEPENDS): deps/%.d: kernel/c/%.c
	$(CC) $(CFLAGS) -MM -MT "deps/$*.d obj/$*.o" $< > $@

ifeq ($(ENABLE_MIPP), 1)
$(CXX_K_DEPENDS): deps/mipp_%.d: kernel/mipp/%.cpp
	$(CXX) $(CXXFLAGS) -MM -MT "deps/mipp_$*.d obj/mipp_$*.o" $< > $@
endif

$(T_DEPENDS): deps/%.d: traces/src/%.c
	$(CC) $(CFLAGS) -MM -MT "deps/$*.d obj/$*.o" $< > $@

$(L_DEPENDS): deps/%.d: obj/%.c
	$(CC) $(CFLAGS) -MM -MT "deps/$*.d obj/$*.o" $< > $@

ifeq ($(ENABLE_CUDA), 1)
$(CUDA_DEPENDS): deps/%.d: src/%.cu
	nvcc $(CUDA_CFLAGS) -MM -MT "deps/$*.d obj/$*.o" $< > $@

$(CUDA_K_DEPENDS): deps/cuda_%.d: kernel/cuda/%.cu
	nvcc $(CUDA_CFLAGS) -MM -MT "deps/cuda_$*.d obj/cuda_$*.o" $< > $@
endif

ifneq ($(MAKECMDGOALS),clean)
-include $(ALL_DEPENDS)
endif

.PHONY: clean
clean: 
	rm -f $(PROGRAM) obj/* deps/*
