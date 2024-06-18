SOURCES = $(wildcard *.cu)
HEADERS = $(wildcard *.h) $(wildcard *.cuh)
BINS = $(SOURCES:.cu=)

# hip
# HIP_PLATFORM=nvidia CUDA_PATH=/usr/lib/nvidia-cuda-toolkit/
CXX = hipcc
GDB = rocgdb
GDBENV := HIP_ENABLE_DEFERRED_LOADING=0 AMD_LOG_LEVEL=3 HSA_FORCE_FINE_GRAIN_PCIE=1
GDBINIT := $(GDB)init
GPUFLAGS := -ggdb -O0 -gmodules --offload-arch=gfx1101 -Rpass-analysis=kernel-resource-usage

# cuda: (oneliner) CXX=nvcc GDB=cuda-gdb GPUFLAGS="-g -G -arch=sm_86"
# CXX = nvcc
# GDB = cuda-gdb
# GDBINIT := cuda-gdbinit
# GPUFLAGS := -g -G -arch=sm_86

NDIRS := 33

$(BINS) : $(SOURCES) $(HEADERS)
	$(CXX) $(GPUFLAGS) $@.cu -o $@

dbg_fee : fee
	$(GDBENV) $(GDB) -x $(GDBINIT) --args ./$< $(NDIRS)
dbg_p1sin : p1sin
	$(GDBENV) $(GDB) -x $(GDBINIT) --args ./$< $(NDIRS)
dbg_crux : crux
	$(GDBENV) $(GDB) -x $(GDBINIT) --args ./$< $(NDIRS)

clean:
	rm -f ./fee ./p1sin
