SOURCES = $(wildcard *.cu)
BINS = $(SOURCES:.cu=)

.PHONY: test

CXX = hipcc
GDB = rocgdb
GDBINIT := rocgdbinit
CXXFLAGS := -g -O0 -gmodules
GPUFLAGS := --offload-arch=gfx1101
NDIRS := 33

$(BINS) :
	$(CXX) $(CXXFLAGS) $(GPUFLAGS) $@.cu -o $@

fee_debug : fee
	$(GDB) -x $(GDBINIT) --args ./fee $(NDIRS)
p1sin_debug : p1sin
	$(GDB) -x $(GDBINIT) --args ./p1sin $(NDIRS)

clean:
	rm -f ./fee ./p1sin