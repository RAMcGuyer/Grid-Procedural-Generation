// source: https://github.com/sunnlo/BellmanFord/blob/master/cuda_bellman_ford.cu
#include <stdio.h>

__device__ int calcStep(int n1, int n2){
     int result = n2 - n1; //Check relative position of the coordinate
     if(result < 0){result = -1;} //destination is less then src, thus the negative step
     else if(result > 0){result = 1;}

    return result;
}
/*getPaths
* Each threads fill a portions of routesX and routesY. The begin at the coordinates of srcsX[tid] and srcsY[tid]
*   then then move allong the coordiante grid in a stair pattern (right-up-right etc) until they reach the
    coordinates associated with destsX[tid] and destsY[tid]. 
*/
__global__ void getPaths(int totalSize, int* routesX, int* routesY, int* srcsX, int* srcsY, int* destsX, int* destsY, int* sizes){
    int tid = threadIdx.x + blockIdx.x*blockDim.x; //Num threads path_sz - 1
    int start;
    int end;
    int x = srcsX[tid];
    int y = srcsY[tid];
    int xBound = destsX[tid];
    int yBound = destsY[tid];
    int xStep = calcStep(x, xBound);
    int yStep = calcStep(y, yBound);
    int stepType = 0;
    if(tid == 0){
        start = 0;
        end = sizes[tid];
    }
    else{
        start = sizes[tid-1];
        end = sizes[tid];
    }
    
    for(int i = start; i < end; i++){
        stepType = i%2;
        routesX[i] = x;
	routesY[i] = y;
	if(stepType == 0){
	    if(x != xBound){
		x += xStep;
            }
            else if(y != yBound){
                y += yStep; 
            }
        }
        else{
            if(y != yBound){
                y += yStep;
            }
            else if (x != xBound){
                x += xStep;
            }
        }
           
    }
    __syncthreads();
}    
