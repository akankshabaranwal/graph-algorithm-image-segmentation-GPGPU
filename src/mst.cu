//
// Created by gyorgy on 16/11/2020.
//

#include "mst.h"

#define CHANNEL_SIZE 3
#define K 50

// Kernel to encode graph
__global__
void encode(char *image, uint4 vertices[], uint3 edges[]) {
}

// Kernel to decode graph
__global__
void decode(uint4 vertices[], char *image) {

}

// Kernel to find min edge
__global__
void find_min_edges(uint4 vertices[], uint3 edges[], uint2 min_edges[], uint num_components) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;

}

// Kernel to remove cycles
__global__
void remove_cycles(uint2 min_edges[], uint num_components) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;

}

// Kernel to merge components
__global__
void merge(uint4 vertices[], uint3 edges[], uint2 min_edges[], uint k, uint num_components) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;

}

// Kernel to orchestrate
__global__
void segment(uint4 vertices[], uint3 edges[], uint2 min_edges[], uint k) {

}

char *compute_segments(void *input, uint x, uint y, uint k) {
    uint4 *vertices;
    uint3 *edges;
    uint2 *min_edges;

    cudaMalloc(&vertices, x*y*sizeof(uint4));
    cudaMalloc(&edges, x*y*sizeof(uint3));
    cudaMalloc(&min_edges, x*y*sizeof(uint2)); // max(min_edges) == vertices.length

    // Write to the matrix from image
    encode<<<1, 1>>>((char*)input, vertices, edges);

    // Segment matrix
    segment<<<1, 1>>>(vertices, edges, min_edges, x*y);

    // Write image back from segmented matrix
    decode<<<1, 1>>>(vertices, (char*)input);

    // Clean up matrix
    cudaFree(vertices);
    cudaFree(edges);
    cudaFree(min_edges);

    //Copy image data back from GPU
    char *output = (char*) malloc(x*y*CHANNEL_SIZE*sizeof(char));

    cudaMemcpy(output, input, x*y*CHANNEL_SIZE*sizeof(char), cudaMemcpyDeviceToHost);

    // Clean up image
    cudaFree(input);

    return output;
}
