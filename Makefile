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

LC_GEOMETRY_HEADERS = lccube_geometry.h lightcone_geometry.hpp
lightcone-raytracer tests/lightcone-tests tests/lightcone-storage-tests tests/lightcone-time-tests tests/lightcone-reference-tests validation/validation-tools: $(LC_GEOMETRY_HEADERS)

$(TARGET): $(SRC)
	$(NVCC) $(OPT_FLAGS) -std=$(CXXSTD) $(GENCODE) $(OMP_FLAGS) $(HDF5_CFLAGS) $(INCLUDES) -o $@ $< $(HDF5_LIBS) -lchealpix $(LIBS)

lightcone-raytracer: lightcone_maps.hpp lightcone_maps.cpp lightcone_layout.cpp lightcone-raytracer.cu lightcone.cpp lightcone.hpp lightcone_sampling.hpp lightcone_ray.hpp
	$(NVCC) $(OPT_FLAGS) -std=$(CXXSTD) $(GENCODE) $(OMP_FLAGS) $(HDF5_CFLAGS) $(INCLUDES) -o $@ lightcone-raytracer.cu lightcone.cpp lightcone_layout.cpp lightcone_maps.cpp $(HDF5_LIBS) -lchealpix $(LIBS)

tests/lightcone-tests: tests/lightcone-tests.cpp lightcone_layout.cpp lightcone.cpp lightcone.hpp lightcone_sampling.hpp lightcone_ray.hpp
	$(CXX) -O2 -Wall -Wextra -Wno-unknown-pragmas -std=$(CXXSTD) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ tests/lightcone-tests.cpp lightcone.cpp lightcone_layout.cpp $(HDF5_LIBS) $(LIBS)

tests/snapshot-reference.o: tests/snapshot-reference.cu tests/snapshot-reference.hpp snapshot-raytracer.cu
	$(NVCC) $(OPT_FLAGS) -std=$(CXXSTD) $(GENCODE) $(OMP_FLAGS) $(HDF5_CFLAGS) $(INCLUDES) -c -o $@ $<

tests/lightcone-reference-tests: lightcone_layout.cpp tests/lightcone-reference-tests.cpp tests/snapshot-reference.o lightcone.cpp lightcone.hpp lightcone_sampling.hpp lightcone_ray.hpp
	$(NVCC) $(OPT_FLAGS) -std=$(CXXSTD) $(OMP_FLAGS) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ tests/lightcone-reference-tests.cpp lightcone.cpp lightcone_layout.cpp tests/snapshot-reference.o $(HDF5_LIBS) -lchealpix $(LIBS)

validation/validation-tools: lightcone_maps.hpp lightcone_maps.cpp lightcone_layout.cpp validation/validation-tools.cpp lightcone.cpp lightcone.hpp lightcone_sampling.hpp lightcone_ray.hpp
	$(CXX) -O2 -std=$(CXXSTD) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ validation/validation-tools.cpp lightcone.cpp lightcone_layout.cpp lightcone_maps.cpp $(HDF5_LIBS) -lchealpix $(LIBS)

tests/lightcone-storage-tests: tests/lightcone-storage-tests.cpp tests/test_support.hpp lightcone.cpp lightcone_layout.cpp lightcone_maps.cpp lightcone.hpp lightcone_maps.hpp lightcone_sampling.hpp lightcone_ray.hpp
	$(CXX) -O2 -Wall -Wextra -Wno-unknown-pragmas -fopenmp -std=$(CXXSTD) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ tests/lightcone-storage-tests.cpp lightcone.cpp lightcone_layout.cpp lightcone_maps.cpp $(HDF5_LIBS) -lchealpix $(LIBS)

tests/lightcone-time-tests: tests/lightcone-time-tests.cpp tests/test_support.hpp lightcone.cpp lightcone_layout.cpp lightcone.hpp lightcone_sampling.hpp lightcone_ray.hpp
	$(CXX) -O2 -Wall -Wextra -Wno-unknown-pragmas -std=$(CXXSTD) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ tests/lightcone-time-tests.cpp lightcone.cpp lightcone_layout.cpp $(HDF5_LIBS) $(LIBS)

tests/lightcone-geometry-tests: tests/lightcone-geometry-tests.cpp tests/test_support.hpp lightcone.cpp lightcone_layout.cpp lightcone.hpp lightcone_sampling.hpp lightcone_ray.hpp $(LC_GEOMETRY_HEADERS)
	$(CXX) -O2 -Wall -Wextra -Wno-unknown-pragmas -std=$(CXXSTD) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ tests/lightcone-geometry-tests.cpp lightcone.cpp lightcone_layout.cpp $(HDF5_LIBS) $(LIBS)

tests/lightcone-geometry-gpu: tests/lightcone-geometry-gpu.cu lightcone.hpp lightcone_sampling.hpp $(LC_GEOMETRY_HEADERS)
	$(NVCC) $(OPT_FLAGS) -std=$(CXXSTD) $(GENCODE) $(INCLUDES) -I. -o $@ $<

test-gpu: tests/lightcone-geometry-gpu
	./tests/lightcone-geometry-gpu

test-cpu: tests/lightcone-tests tests/lightcone-storage-tests tests/lightcone-time-tests tests/lightcone-geometry-tests
	./tests/lightcone-tests
	./tests/lightcone-geometry-tests
	OMP_NUM_THREADS=4 ./tests/lightcone-storage-tests
	./tests/lightcone-time-tests

test-reference: tests/lightcone-reference-tests
	./tests/lightcone-reference-tests

test-validation: lightcone-raytracer validation/validation-tools
	python3 validation/test_workflow.py

test: test-cpu test-reference test-validation

clean:
	rm -f $(TARGET) lightcone-raytracer tests/lightcone-tests tests/lightcone-reference-tests tests/snapshot-reference.o tests/lightcone-storage-tests tests/lightcone-time-tests tests/lightcone-geometry-tests tests/lightcone-geometry-gpu validation/validation-tools

.PHONY: all clean test test-cpu test-gpu test-reference test-validation
