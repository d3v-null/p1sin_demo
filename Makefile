SOURCES = $(wildcard *.cu)
HEADERS = $(wildcard *.h)
BINS = $(SOURCES:.cu=)

# hip
CXX = hipcc
GDB = rocgdb
GDBINIT := $(GDB)init
GPUFLAGS := --offload-arch=gfx1101 -g -O0 -gmodules

# cuda: (oneliner) CXX=nvcc GDB=cuda-gdb GPUFLAGS="-g -G -arch=sm_86"
# CXX = nvcc
# GDB = cuda-gdb
# GDBINIT := cuda-gdbinit
# GPUFLAGS := -g -G -arch=sm_86

NDIRS := 33

$(BINS) : $(SOURCES) $(HEADERS)
	$(CXX) $(GPUFLAGS) $@.cu -o $@

dbg_fee : fee
	$(GDB) -x $(GDBINIT) --args ./$< $(NDIRS)
dbg_p1sin : p1sin
	$(GDB) -x $(GDBINIT) --args ./$< $(NDIRS)

clean:
	rm -f ./fee ./p1sin