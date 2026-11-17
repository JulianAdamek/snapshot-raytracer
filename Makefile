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
CORRECT_SHEAR_TRANSPORT ?= 1
CORRECT_SACHS_B ?= 1
NONLINEAR_SCALAR ?= 1
CORRECT_OBSERVER_DISTANCE ?= 1
PHYSICS_FLAGS = -DRT_CORRECT_SHEAR_TRANSPORT=$(CORRECT_SHEAR_TRANSPORT) -DRT_CORRECT_SACHS_B=$(CORRECT_SACHS_B) -DRT_NONLINEAR_SCALAR=$(NONLINEAR_SCALAR) -DRT_CORRECT_OBSERVER_DISTANCE=$(CORRECT_OBSERVER_DISTANCE)

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

# Changing a correction switch must rebuild existing binaries and reference objects.
.PHONY: FORCE
.build/physics-config: FORCE
	@mkdir -p .build
	@printf '%s\n' '$(PHYSICS_FLAGS) $(CPPFLAGS) $(OPT_FLAGS)' > .build/physics-config.tmp
	@cmp -s .build/physics-config.tmp $@ || cp .build/physics-config.tmp $@

$(TARGET) lightcone-raytracer tests/snapshot-reference.o tests/lightcone-tests tests/lightcone-storage-tests tests/lightcone-time-tests tests/lightcone-geometry-tests tests/lightcone-reference-tests tests/lightcone-geometry-gpu validation/validation-tools: raytracer_physics.hpp .build/physics-config

LC_GEOMETRY_HEADERS = lccube_geometry.h lightcone_geometry.hpp
lightcone-raytracer tests/lightcone-tests tests/lightcone-storage-tests tests/lightcone-time-tests tests/lightcone-reference-tests validation/validation-tools: $(LC_GEOMETRY_HEADERS)

$(TARGET): $(SRC)
	$(NVCC) $(OPT_FLAGS) -std=$(CXXSTD) $(GENCODE) $(OMP_FLAGS) $(PHYSICS_FLAGS) $(CPPFLAGS) $(HDF5_CFLAGS) $(INCLUDES) -o $@ $< $(HDF5_LIBS) -lchealpix $(LIBS)

lightcone-raytracer: lightcone_preload.hpp lightcone_preload.cpp lightcone_parts.hpp lightcone_parts.cpp lightcone_maps.hpp lightcone_maps.cpp lightcone_layout.cpp lightcone-raytracer.cu lightcone.cpp lightcone.hpp lightcone_sampling.hpp lightcone_ray.hpp
	$(NVCC) $(OPT_FLAGS) -std=$(CXXSTD) $(GENCODE) $(OMP_FLAGS) -Xcompiler -pthread $(PHYSICS_FLAGS) $(CPPFLAGS) $(HDF5_CFLAGS) $(INCLUDES) -o $@ lightcone-raytracer.cu lightcone.cpp lightcone_layout.cpp lightcone_maps.cpp lightcone_preload.cpp lightcone_parts.cpp $(HDF5_LIBS) -lchealpix -lcfitsio $(LIBS)

tests/lightcone-tests: tests/lightcone-tests.cpp lightcone_layout.cpp lightcone.cpp lightcone.hpp lightcone_sampling.hpp lightcone_ray.hpp
	$(CXX) $(PHYSICS_FLAGS) $(CPPFLAGS) -O2 -Wall -Wextra -Wno-unknown-pragmas -std=$(CXXSTD) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ tests/lightcone-tests.cpp lightcone.cpp lightcone_layout.cpp $(HDF5_LIBS) $(LIBS)

tests/snapshot-reference.o: tests/snapshot-reference.cu tests/snapshot-reference.hpp snapshot-raytracer.cu
	$(NVCC) $(OPT_FLAGS) -std=$(CXXSTD) $(GENCODE) $(OMP_FLAGS) $(PHYSICS_FLAGS) $(CPPFLAGS) $(HDF5_CFLAGS) $(INCLUDES) -c -o $@ $<

tests/lightcone-reference-tests: lightcone_layout.cpp tests/lightcone-reference-tests.cpp tests/snapshot-reference.o lightcone.cpp lightcone.hpp lightcone_sampling.hpp lightcone_ray.hpp
	$(NVCC) $(OPT_FLAGS) -std=$(CXXSTD) $(OMP_FLAGS) $(PHYSICS_FLAGS) $(CPPFLAGS) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ tests/lightcone-reference-tests.cpp lightcone.cpp lightcone_layout.cpp tests/snapshot-reference.o $(HDF5_LIBS) -lchealpix $(LIBS)

validation/validation-tools: lightcone_parts.hpp lightcone_parts.cpp lightcone_maps.hpp lightcone_maps.cpp lightcone_layout.cpp validation/validation-tools.cpp lightcone.cpp lightcone.hpp lightcone_sampling.hpp lightcone_ray.hpp
	$(CXX) $(PHYSICS_FLAGS) $(CPPFLAGS) -O2 -std=$(CXXSTD) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ validation/validation-tools.cpp lightcone.cpp lightcone_layout.cpp lightcone_maps.cpp lightcone_parts.cpp $(HDF5_LIBS) -lchealpix -lcfitsio $(LIBS)

tests/lightcone-storage-tests: tests/lightcone-storage-tests.cpp tests/test_support.hpp lightcone.cpp lightcone_layout.cpp lightcone_maps.cpp lightcone.hpp lightcone_maps.hpp lightcone_sampling.hpp lightcone_ray.hpp
	$(CXX) $(PHYSICS_FLAGS) $(CPPFLAGS) -O2 -Wall -Wextra -Wno-unknown-pragmas -fopenmp -std=$(CXXSTD) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ tests/lightcone-storage-tests.cpp lightcone.cpp lightcone_layout.cpp lightcone_maps.cpp $(HDF5_LIBS) -lchealpix $(LIBS)

tests/lightcone-time-tests: tests/lightcone-time-tests.cpp tests/test_support.hpp lightcone.cpp lightcone_layout.cpp lightcone.hpp lightcone_sampling.hpp lightcone_ray.hpp
	$(CXX) $(PHYSICS_FLAGS) $(CPPFLAGS) -O2 -Wall -Wextra -Wno-unknown-pragmas -std=$(CXXSTD) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ tests/lightcone-time-tests.cpp lightcone.cpp lightcone_layout.cpp $(HDF5_LIBS) $(LIBS)

tests/lightcone-geometry-tests: tests/lightcone-geometry-tests.cpp tests/test_support.hpp lightcone.cpp lightcone_layout.cpp lightcone.hpp lightcone_sampling.hpp lightcone_ray.hpp $(LC_GEOMETRY_HEADERS)
	$(CXX) $(PHYSICS_FLAGS) $(CPPFLAGS) -O2 -Wall -Wextra -Wno-unknown-pragmas -std=$(CXXSTD) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ tests/lightcone-geometry-tests.cpp lightcone.cpp lightcone_layout.cpp $(HDF5_LIBS) $(LIBS)

tests/lightcone-geometry-gpu: tests/lightcone-geometry-gpu.cu lightcone.hpp lightcone_sampling.hpp $(LC_GEOMETRY_HEADERS)
	$(NVCC) $(OPT_FLAGS) -std=$(CXXSTD) $(GENCODE) $(PHYSICS_FLAGS) $(CPPFLAGS) $(INCLUDES) -I. -o $@ $<

test-gpu-geometry: tests/lightcone-geometry-gpu
	./tests/lightcone-geometry-gpu

test-cpu: tests/lightcone-tests tests/lightcone-storage-tests tests/lightcone-time-tests tests/lightcone-geometry-tests
	./tests/lightcone-tests
	./tests/lightcone-geometry-tests
	OMP_NUM_THREADS=4 ./tests/lightcone-storage-tests
	./tests/lightcone-time-tests

test-reference: tests/lightcone-reference-tests
	./tests/lightcone-reference-tests

tests/lightcone-bundle-tests: tests/lightcone-bundle-tests.cpp tests/analytic_fields.hpp tests/test_support.hpp lightcone.cpp lightcone_layout.cpp lightcone_ray.hpp lightcone_sampling.hpp lightcone.hpp $(LC_GEOMETRY_HEADERS) raytracer_physics.hpp .build/physics-config
	$(CXX) $(PHYSICS_FLAGS) $(CPPFLAGS) -O2 -Wall -Wextra -Wno-unknown-pragmas -std=$(CXXSTD) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ tests/lightcone-bundle-tests.cpp lightcone.cpp lightcone_layout.cpp $(HDF5_LIBS) $(LIBS)

# Named objects avoid ODR mismatches and permit all four historical error controls
# to coexist. Hold scalar/observer corrections fixed for this 2x2 test matrix.
define ODE_VARIANT
.build/variant-$(1)$(2)$(3)$(4).o: tests/snapshot-reference.cu snapshot-raytracer.cu raytracer_physics.hpp tests/snapshot-reference.hpp
	@mkdir -p .build
	$$(NVCC) $$(OPT_FLAGS) -std=$$(CXXSTD) $$(GENCODE) $$(OMP_FLAGS) -DRT_CORRECT_SHEAR_TRANSPORT=$(1) -DRT_CORRECT_SACHS_B=$(2) -DRT_NONLINEAR_SCALAR=$(if $(3),$(3),1) -DRT_CORRECT_OBSERVER_DISTANCE=$(if $(4),$(4),1) $$(HDF5_CFLAGS) $$(INCLUDES) -c -o $$@ $$<
.build/variant-$(1)$(2)$(3)$(4): tests/ode-variant-tests.cpp .build/variant-$(1)$(2)$(3)$(4).o tests/test_support.hpp lightcone.cpp lightcone_layout.cpp lightcone.hpp lightcone_ray.hpp lightcone_sampling.hpp raytracer_physics.hpp $$(LC_GEOMETRY_HEADERS)
	$$(NVCC) $$(OPT_FLAGS) -std=$$(CXXSTD) $$(OMP_FLAGS) -DRT_CORRECT_SHEAR_TRANSPORT=$(1) -DRT_CORRECT_SACHS_B=$(2) -DRT_NONLINEAR_SCALAR=$(if $(3),$(3),1) -DRT_CORRECT_OBSERVER_DISTANCE=$(if $(4),$(4),1) $$(HDF5_CFLAGS) $$(INCLUDES) -I. -o $$@ tests/ode-variant-tests.cpp lightcone.cpp lightcone_layout.cpp .build/variant-$(1)$(2)$(3)$(4).o $$(HDF5_LIBS) -lchealpix $$(LIBS)
endef
$(foreach s,0 1,$(foreach b,0 1,$(eval $(call ODE_VARIANT,$(s),$(b)))))
$(eval $(call ODE_VARIANT,0,0,0,0))
test-variants: .build/variant-0000 .build/variant-00 .build/variant-01 .build/variant-10 .build/variant-11
	./.build/variant-0000
	./.build/variant-00
	./.build/variant-01
	./.build/variant-10
	./.build/variant-11

test-bundle: tests/lightcone-bundle-tests
	./tests/lightcone-bundle-tests

test-validation: lightcone-raytracer validation/validation-tools
	python3 validation/test_workflow.py
	python3 validation/test_ode_comparison.py
	python3 validation/test_campaign.py

test: test-cpu test-reference test-validation test-bundle test-variants

clean:
	rm -f $(TARGET) lightcone-raytracer tests/lightcone-tests tests/lightcone-reference-tests tests/snapshot-reference.o tests/lightcone-storage-tests tests/lightcone-time-tests tests/lightcone-geometry-tests tests/lightcone-geometry-gpu tests/lightcone-bundle-tests validation/validation-tools
	rm -f tests/lightcone-preload-tests tests/lightcone-preload-gpu tests/lightcone-preload-gpu-legacy
	rm -rf .build

.PHONY: all clean test test-cpu test-gpu test-reference test-validation test-bundle test-variants

tests/lightcone-preload-tests: tests/lightcone-preload-tests.cpp tests/test_support.hpp lightcone_preload.cpp lightcone_preload.hpp lightcone_parts.cpp lightcone_parts.hpp lightcone.cpp lightcone_layout.cpp lightcone_maps.cpp lightcone_maps.hpp lightcone.hpp $(LC_GEOMETRY_HEADERS) raytracer_physics.hpp .build/physics-config
	$(CXX) $(PHYSICS_FLAGS) $(CPPFLAGS) -O2 -Wall -Wextra -Wno-unknown-pragmas -std=$(CXXSTD) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ tests/lightcone-preload-tests.cpp lightcone_preload.cpp lightcone_parts.cpp lightcone.cpp lightcone_layout.cpp lightcone_maps.cpp $(HDF5_LIBS) -lchealpix -lcfitsio $(LIBS)

test-preload: tests/lightcone-preload-tests
	./tests/lightcone-preload-tests

test: test-preload
.PHONY: test-preload

LC_PRELOAD_SOURCES = lightcone.cpp lightcone_layout.cpp lightcone_maps.cpp lightcone_preload.cpp lightcone_parts.cpp
LC_PRELOAD_HEADERS = lightcone_preload.hpp lightcone_parts.hpp lightcone_maps.hpp lightcone.hpp lightcone_sampling.hpp lightcone_ray.hpp $(LC_GEOMETRY_HEADERS) raytracer_physics.hpp

tests/lightcone-preload-gpu: tests/lightcone-preload-gpu.cu tests/test_support.hpp lightcone-raytracer.cu $(LC_PRELOAD_SOURCES) $(LC_PRELOAD_HEADERS) .build/physics-config
	$(NVCC) $(OPT_FLAGS) -std=$(CXXSTD) $(GENCODE) $(OMP_FLAGS) -Xcompiler -pthread $(PHYSICS_FLAGS) $(CPPFLAGS) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ $< $(LC_PRELOAD_SOURCES) $(HDF5_LIBS) -lchealpix -lcfitsio $(LIBS)

tests/lightcone-preload-gpu-legacy: tests/lightcone-preload-gpu.cu tests/test_support.hpp lightcone-raytracer.cu $(LC_PRELOAD_SOURCES) $(LC_PRELOAD_HEADERS)
	$(NVCC) $(OPT_FLAGS) -std=$(CXXSTD) $(GENCODE) $(OMP_FLAGS) -Xcompiler -pthread -DRT_CORRECT_SHEAR_TRANSPORT=0 -DRT_CORRECT_SACHS_B=0 -DRT_NONLINEAR_SCALAR=0 -DRT_CORRECT_OBSERVER_DISTANCE=0 $(CPPFLAGS) $(HDF5_CFLAGS) $(INCLUDES) -I. -o $@ $< $(LC_PRELOAD_SOURCES) $(HDF5_LIBS) -lchealpix -lcfitsio $(LIBS)

test-gpu: test-gpu-geometry tests/lightcone-preload-gpu tests/lightcone-preload-gpu-legacy
	./tests/lightcone-preload-gpu
	./tests/lightcone-preload-gpu-legacy

.PHONY: test-gpu-geometry
