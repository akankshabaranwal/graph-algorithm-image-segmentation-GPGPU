//
// Created by gyorgy on 16/11/2020.
//

#include "mst.h"

#define CHANNEL_SIZE 3
#define K 50
#define NUM_NEIGHBOURS 4

/*
 * Matrix structure:
 *      - vertices array of type uint4, where
 *          * x = id
 *          * y = component id
 *          * z = component size
 *          * w = component internal difference
 *
 *      - edges 2D array of type uint3, where
 *          * x = destination id
 *          * y = weight (dissimilarity)
 *          * z = component id
 *
 * Min edges:
 *      - x = weight
 *      - y = source id
 *      - z = destination id
 */


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
void find_min_edges(uint4 vertices[], uint3 edges[], uint3 min_edges[], uint num_components) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_components) return;
    uint3 min;
    min.x = UINT_MAX;
    // Scan all vertices and find the min with component == tid
    for (int i = 0; i < num_components; i++) {
        uint4 vertice = vertices[i];
        if (vertice.y == tid) {
            for (int j = tid * NUM_NEIGHBOURS; j < tid * NUM_NEIGHBOURS + NUM_NEIGHBOURS; j++) {
                uint3 edge = edges[j];
                if (edge.x != 0) {
                    if (edge.y < min.x) {
                        min.x = edge.y;
                        min.y = vertice.x;
                        min.z = edge.x;
                    }
                }
            }
        }
    }
    min_edges[tid] = min;
}

// Kernel to remove cycles
__global__
void remove_cycles(uint3 min_edges[], uint num_components) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;

}

// Kernel to merge components
__global__
void merge(uint4 vertices[], uint3 edges[], uint3 min_edges[], uint *num_components) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;

}

// Kernel to orchestrate
__global__
void segment(uint4 vertices[], uint3 edges[], uint3 min_edges[], uint n_components) {
    uint prev_n_components = 0;
    while (n_components != prev_n_components) {
        prev_n_components = n_components;
        uint threads = 1024;
        uint blocks = 1;
        if (n_components < 1024) {
            threads = n_components;
        } else {
            blocks = n_components / 1024 + 1;
        }
        find_min_edges<<<threads, blocks>>>(vertices, edges, min_edges, n_components);
        cudaDeviceSynchronize();
        __syncthreads();
        remove_cycles<<<threads, blocks>>>(min_edges, n_components);
        cudaDeviceSynchronize();
        __syncthreads();
        merge<<<threads, blocks>>>(vertices, edges, min_edges, &n_components);
        cudaDeviceSynchronize();
        __syncthreads();
    }
}

char *compute_segments(void *input, uint x, uint y) {
    uint4 *vertices;
    uint3 *edges;
    uint3 *min_edges;

    cudaMalloc(&vertices, x*y*sizeof(uint4));
    cudaMalloc(&edges, x*y*sizeof(uint3));
    cudaMalloc(&min_edges, x*y*sizeof(uint3)); // max(min_edges) == vertices.length

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
