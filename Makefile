SOURCES = $(wildcard *.cu)
OBJECTS = $(SOURCES:.cu=.o)
BINS = $(SOURCES:.cu=)

.PHONY: test

CXX = hipcc
CXXFLAGS = -g -O0 -gmodules
HIPFLAGS = --offload-arch=gfx1101

%.o: %.cu
	$(CXX) $(CCFLAGS) -c $< -o $@

./fee : fee.o
	$(CXX) fee.o -o $@
./p1sin : p1sin.o
	$(CXX) p1sin.o -o $@

clean:
	rm -f $(BINS )
	rm -f $(OBJECTS)