// source: https://github.com/sunnlo/BellmanFord/blob/master/cuda_bellman_ford.cu
#include <stdio.h>
#include "Coord2D.h"

__global__ void bellman_ford_kernel(int n, int* d_edges, int* d_distances, bool* next) {
    unsigned INF = 1000000;
    int thread = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // range check
    if(thread >= n) {return;}

    // every edge in d_edges is updated by a thread
    for(int i = 0; i < n; ++i) {
        for(int t = thread; t < n; t += stride) {
            int weight = d_edges[i*n+t];
            if(weight < INF) {
                int new_dist = d_distances[i] + weight;

                // update distance at t if new min found
                if(new_dist < d_distances[t]) {
                    d_distances[t] = new_dist;
                    *next = true; // return true through pointer param
                }
            }
        }
    }
}

/**
    @param blocksPerGrid
    @param theadsPerBlock
    @param n - number of vertices
    @param edges - (n*n)-sized array of edges; for a graph with n vertices, there are a maximum of n*(n-1) edges
    @param distances - array of distances from source vertex to other vertices in graph
**/
void bellman_ford(int blocksPerGrid, int threadsPerBlock, int n, int* edges, int* distances) {
    unsigned INF = 1000000;
    dim3 dim_grid(blocksPerGrid);
    dim3 dim_block(threadsPerBlock);

    int* d_edges;
    int* d_distances;

    // denote whether there should be another iteration of bellman-ford kernel
    bool* d_next; 
    bool* h_next; 

    // Allocate device variables
    cudaMalloc(&d_edges, sizeof(int)*n*n);
    cudaMalloc(&d_distances, sizeof(int)*n);
    cudaMalloc(&d_next, sizeof(bool));

    // initialize distances to source to INF
    for(int i = 0; i < n; ++i) {
        distances[i] = INF;
    }
    distances[0] = 0; // dist from source to itself

    // Copy host variables to device
    cudaMemcpy(d_edges, edges, sizeof(int)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_distances, distances, sizeof(int)*n, cudaMemcpyHostToDevice);

    for(;;) {
        // initialize next iteration to false
        *h_next = false;
        cudaMemcpy(d_next, h_next, sizeof(bool), cudaMemcpyHostToDevice);

        bellman_ford_kernel<<<dim_grid, dim_block>>> (n, d_edges, d_distances, d_next);
        cudaDeviceSynchronize();

        // Copy result of next back to host
        cudaMemcpy(h_next, d_next, sizeof(bool), cudaMemcpyDeviceToHost);
        if(!(*h_next)) {
            break;
        }
    }
    // Copy distances array back to host
    cudaMemcpy(distances, d_distances, sizeof(int)*n, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_edges);
    cudaFree(d_distances);
    cudaFree(d_next);
}


 __global__ void calcPaths(Coord2D* routes, int start, int end, int index, Coord2D* srcs, Coord2D* dests){

    int xStep = dests[index].first - srcs[index].first;
    int yStep = dests[index].second - srcs[index].second;

    if(xStep < 0) {xStep /= (-1)*xStep;}
    else if(xStep > 0) {xStep /= xStep;}
    if(yStep < 0) {yStep /= (-1)*yStep;}
    else if(yStep > 0) {yStep /= yStep;}
    routes[start].first = xStep;
    routes[start].second = yStep;
    printf("Routes at start: first=%d second=%d\n", routes[start].first, routes[start].second);
}

__global__ void getPaths(Coord2D* routes, int* sizes, Coord2D* srcs, Coord2D* dests){
    int tid = threadIdx.x + blockIdx.x*blockDim.x; //Num threads path_sz - 1
    printf("Inside kernel with thread: %d\n", tid);
    int start = sizes[tid];
    int end = sizes[tid+1];
    calcPaths<<<1, 1>>> (routes, start, end, tid, srcs, dests);
    cudaDeviceSynchronize();
}
