CC	:= g++
CXXFLAGS:= -std=c++11 #-Wall -Wno-nonnull-compare #-Wfatal-errors
LDFLAGS := 

TARGETS := procgen
OBJ	:= MainDriver.o Coord2D.o Tile.o Path.o Grid2D.o GameGrid2D.o
DEPS	:= Coord2D.h Tile.h Path.h Grid2D.h GameGrid2D.h #Hash.h

.PHONY: all clean

all: $(TARGETS)

clean:
	rm -f $(TARGETS) $(OBJ)

cuda: MainDriver.cpp Path.cpp Coord2D.cpp Tile.cpp Grid2D.cpp GameGrid2D.cu kernel.cu Path.h Coord2D.h Tile.h Grid2D.h GameGrid2D.h
	nvcc -std=c++11 MainDriver.cpp GameGrid2D.cu Coord2D.cpp Grid2D.cpp Path.cpp Tile.cpp -lcudart -arch=sm_35  -L/usr/local/cuda/lib64 -o cudaprocgen

$(OBJ): %.o : %.cpp $(DEPS)
	$(CC) -g -c -o $@ $< $(CXXFLAGS)

$(TARGETS): $(OBJ)
	$(CC) -o $@ $(LIBS)$^ $(CXXFLAGS) $(LDFLAGS)
	rm -f $(OBJ)

