CC	= g++
NVCC = nvcc
CXX_FLAGS = -std=c++11#-Wall -Wno-nonnull-compare #-Wfatal-errors
NVCC_FLAGS = -O3 -I/usr/local/cuda/include -arch=sm_30
LD_FLAGS = -lcudart -L/usr/local/cuda/lib64
EXE = procgen
OBJ	= MainDriver.o Coord2D.o Tile.o Path.o Grid2D.o
DEPS	= Coord2D.h Tile.h Path.h Grid2D.h GameGrid2D.h#Hash.h

CUDA_OBJ := GameGrid2D.o
CUDA_DEPS := GameGrid2D.cu kernel.cu

.PHONY: all clean

all: $(EXE)

$(OBJ): %.o : %.cpp $(DEPS)
	$(NVCC) -g -c -o $@ $< $(CXX_FLAGS) $(NVCC_FLAGS)

$(CUDA_OBJ): %.o : %.cu $(CUDA_DEPS)
	$(NVCC) -g -c -o $@ $< $(CXX_FLAGS) $(NVCC_FLAGS)

$(EXE): $(OBJ) $(CUDA_OBJ)
	$(NVCC) $(OBJ) $(CUDA_OBJ) -o $(EXE) $(LD_FLAGS)
	rm -f $(OBJ) $(CUDA_OBJ)

clean:
	rm -rf $(EXE) $(OBJ) $(CUDA_OBJ)
