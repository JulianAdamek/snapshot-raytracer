NVCC ?= nvcc
TARGET ?= snapshot-raytracer
SRC ?= snapshot-raytracer.cu

# ARCH options: rtx3060, a100, grace_hopper
ARCH ?= rtx3060

CXXSTD ?= c++17
OPT_FLAGS ?= -O3 --use_fast_math
OMP_FLAGS ?= -Xcompiler -fopenmp

# Try pkg-config for HDF5; allow overrides from the environment.
HDF5_CFLAGS ?= $(shell pkg-config --cflags hdf5 2>/dev/null)
HDF5_LIBS ?= $(shell pkg-config --libs hdf5 2>/dev/null) -lhdf5_hl

# Optional extra include/lib flags can be passed in from the environment.
INCLUDES ?=
LIBS ?=

ifeq ($(ARCH),rtx3060)
GENCODE = -gencode arch=compute_86,code=sm_86
else ifeq ($(ARCH),a100)
GENCODE = -gencode arch=compute_80,code=sm_80
else ifeq ($(ARCH),grace_hopper)
GENCODE = -gencode arch=compute_90,code=sm_90a
else
$(error Unknown ARCH '$(ARCH)'. Use rtx3060, a100, or grace_hopper)
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(OPT_FLAGS) -std=$(CXXSTD) $(GENCODE) $(OMP_FLAGS) $(HDF5_CFLAGS) $(INCLUDES) -o $@ $< $(HDF5_LIBS) -lchealpix $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
