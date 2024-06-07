SOURCES = $(wildcard *.cu)
BINS = $(SOURCES:.cu=)

.PHONY: test

CXX = hipcc
GDB = rocgdb
GDBINIT := rocgdbinit
CXXFLAGS := -g -O0 -gmodules
GPUFLAGS := --offload-arch=gfx1101

$(BINS) :
	$(CXX) $(CCFLAGS) $(GPUFLAGS) $@.cu -o $@

fee_debug : fee
	$(GDB) -x $(GDBINIT) --args ./fee
p1sin_debug : p1sin
	$(GDB) -x $(GDBINIT) --args ./p1sin

clean:
	rm -f ./fee ./p1sin