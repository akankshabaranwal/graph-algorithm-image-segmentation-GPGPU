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
void find_min_edges(uint4 vertices[], uint3 edges[], uint3 min_edges[], uint num_components, uint vertices_length) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_components) return;
    uint3 min;
    min.x = UINT_MAX;
    // Scan all vertices and find the min with component == tid
    for (int i = 0; i < vertices_length; i++) {
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
    if (tid >= num_components) return;

    uint3 edge = min_edges[tid];
    uint src = edge.y;
    uint dest = edge.z;
    __syncthreads();

    for (int i = 0; i < num_components; i++) {
        if (i == tid) continue;
        uint3 curr_edge = min_edges[i];
        if (src == curr_edge.z && dest == curr_edge.y) {
            if (i < tid) return;
            curr_edge.z = dest;
        }
    }

}

// Kernel to merge components
__global__
void merge(uint4 vertices[], uint3 edges[], uint3 min_edges[], uint *num_components, uint vertices_length) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= *num_components) return;
    uint3 min_edge = min_edges[tid];
    uint4 src = vertices[min_edge.y];
    uint4 dest = vertices[min_edge.z];
    __syncthreads();
    uint src_diff = src.w + (K / src.z);
    uint dest_diff = dest.w + (K / dest.z);
    if (min_edge.x <= min(src_diff, dest_diff)) {
        atomicSub_system(num_components, 1); // Is this horribly inefficient?
        uint new_int_diff = max(src.w, max(dest.w, min_edge.x));
        uint new_size = src.z + dest.z;
        uint new_component = src.x;

        for (int i = 0; i < vertices_length; i++) {
            uint4 vertice = vertices[i];
            bool is_vertice_new_comp = vertice.y == dest.y || vertice.y == src.y;
            if (is_vertice_new_comp) {
                vertice.y = new_component;
                vertice.z = new_size;
                vertice.w = new_int_diff;
            }

            for (int j = tid * NUM_NEIGHBOURS; j < tid * NUM_NEIGHBOURS + NUM_NEIGHBOURS; j++) {
                uint3 neighbour_edge = edges[j];
                if (neighbour_edge.x != 0) {
                    if (neighbour_edge.y == dest.y) {
                        if (is_vertice_new_comp) neighbour_edge.x = 0; // Remove internal edges
                        else neighbour_edge.z = new_component;
                    }
                }
            }
        }
    }
}

// Kernel to orchestrate
__global__
void segment(uint4 vertices[], uint3 edges[], uint3 min_edges[], uint *n_components) {
    uint prev_n_components = 0;
    uint n_vertices = *n_components;
    uint curr_n_comp = *n_components;
    while (curr_n_comp != prev_n_components) {
        uint threads = 1024;
        uint blocks = 1;
        if (*n_components < 1024) {
            threads = curr_n_comp;
        } else {
            blocks = curr_n_comp / 1024 + 1;
        }
        find_min_edges<<<blocks, threads>>>(vertices, edges, min_edges, curr_n_comp, n_vertices);
        cudaDeviceSynchronize();
        __syncthreads();
        remove_cycles<<<blocks, threads>>>(min_edges, curr_n_comp);
        cudaDeviceSynchronize();
        __syncthreads();
        merge<<<blocks, threads>>>(vertices, edges, min_edges, n_components, n_vertices);
        cudaDeviceSynchronize();
        __syncthreads();

        prev_n_components = curr_n_comp;
        curr_n_comp = *n_components;
    }
}

char *compute_segments(void *input, uint x, uint y) {
    uint4 *vertices;
    uint3 *edges;
    uint3 *min_edges;
    uint num_vertices = x * y;
    uint *num_vertices_dev;

    cudaMalloc(&vertices, num_vertices*sizeof(uint4));
    cudaMalloc(&edges, num_vertices*sizeof(uint3));
    cudaMalloc(&min_edges, num_vertices*sizeof(uint3)); // max(min_edges) == vertices.length
    cudaMalloc(&num_vertices_dev, sizeof(uint));

    cudaMemcpy(num_vertices_dev, &num_vertices, sizeof(uint), cudaMemcpyHostToDevice);

    // Write to the matrix from image
    encode<<<1, 1>>>((char*)input, vertices, edges);

    // Segment matrix
    segment<<<1, 1>>>(vertices, edges, min_edges, num_vertices_dev);

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
