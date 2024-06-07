SOURCES = fee.cu
OBJECTS = $(SOURCES:.cu=.o)
EXECUTABLE=./fee

.PHONY: test

all: $(EXECUTABLE) test

CXX = hipcc
CXXFLAGS =-g -O0 -gmodules
HIPFLAGS = --offload-arch=gfx1101

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

fee.o:
	$(CXX) $(CXXFLAGS) -c fee.cu

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)