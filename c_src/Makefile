CC		:= g++
CXXFLAGS:= -std=c++11
LDFLAGS := 

TARGETS := procgen
MAINS	:= $(addsuffix .o, $(TARGETS) )
OBJ		:= MainDriver.o Coord2D.o Tile.o Path.o Grid2D.o GameGrid2D.o
DEPS	:= Coord2D.h Tile.h Path.h Grid2D.h GameGrid2D.h Hash.h

.PHONY: all clean

all: $(TARGETS)

clean:
	rm -f $(TARGETS) $(OBJ)

$(OBJ): %.o : %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CXXFLAGS)

$(TARGETS): $(OBJ)
	$(CC) -o $@ $(LIBS)$^ $(CXXFLAGS) $(LDFLAGS)
