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
 *          * x = id (starting at 1, so that 1 can represent null node)
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
    uint component_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (component_id >= num_components) return;

    uint3 min;
    min.x = UINT_MAX;
    // Scan all vertices and find the min with component == tid
    for (int i = 0; i < vertices_length; i++) {
        uint4 vertice = vertices[i];
        if (vertice.y == component_id) {
            for (int j = component_id * NUM_NEIGHBOURS; j < component_id * NUM_NEIGHBOURS + NUM_NEIGHBOURS; j++) {
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
    min_edges[component_id] = min;
}

// Kernel to remove cycles
__global__
void remove_cycles(uint3 min_edges[], uint num_components) {
    uint component_id_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (component_id_x >= num_components) return;

    uint component_id_y = blockDim.y * blockIdx.y + threadIdx.y;
    if (component_id_y >= num_components) return;

    if (component_id_x == component_id_y) return;

    uint3 edge = min_edges[component_id_x];
    uint src = edge.y;
    uint dest = edge.z;
    __syncthreads();

    uint3 curr_edge = min_edges[component_id_y];
    if (src == curr_edge.z && dest == curr_edge.y && component_id_x > component_id_y) {
        curr_edge.z = dest;
    }
}

// Kernel to update vertices with new components
__global__
void update_matrix(uint4 vertices[], uint3 edges[], uint vertices_length, uint new_component, uint new_size, uint new_int_diff, uint dest_id, uint src_id) {
    uint vertice_id = blockDim.y * blockIdx.y + threadIdx.y;
    if (vertice_id >= vertices_length) return;

    uint4 vertice = vertices[vertice_id];
    bool is_vertice_new_comp = vertice.y == dest_id || vertice.y == src_id;
    if (is_vertice_new_comp) {
        vertice.y = new_component;
        vertice.z = new_size;
        vertice.w = new_int_diff;
    }

    for (int j = vertice_id * NUM_NEIGHBOURS; j < vertice_id * NUM_NEIGHBOURS + NUM_NEIGHBOURS; j++) {
        uint3 neighbour_edge = edges[j];
        if (neighbour_edge.x != 0) {
            if (neighbour_edge.y == dest_id) {
                if (is_vertice_new_comp) neighbour_edge.x = 0; // Remove internal edges
                else neighbour_edge.z = new_component;
            }
        }
    }

}

// Kernel to merge components
__global__
void merge(uint4 vertices[], uint3 edges[], uint3 min_edges[], uint *num_components, uint update_threads, uint update_blocks, uint vertices_length) {
    uint component_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (component_id >= *num_components) return;

    uint3 min_edge = min_edges[component_id];
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

        update_matrix<<<update_blocks, update_blocks>>>(vertices, edges, vertices_length, new_component, new_size, new_int_diff, dest.y, src.y);
    }
}

// Kernel to orchestrate
__global__
void segment(uint4 vertices[], uint3 edges[], uint3 min_edges[], uint *n_components) {
    uint prev_n_components = 0;
    uint n_vertices = *n_components;
    uint curr_n_comp = *n_components;
    dim3 threads;
    dim3 blocks;
    if (n_vertices < 1024) {
        threads.y = n_vertices;
        blocks.y = 1;
    } else {
        threads.y = 1024;
        blocks.y = n_vertices / 1024 + 1;
    }

    while (curr_n_comp != prev_n_components) {
        if (curr_n_comp < 1024) {
            threads.x = curr_n_comp;
            blocks.x = 1;
        } else {
            threads.x = 1024;
            blocks.x = curr_n_comp / 1024 + 1;
        }

        dim3 cycle_blocks;
        cycle_blocks.x = blocks.x;
        cycle_blocks.y = blocks.x;
        dim3 cycle_threads;
        cycle_threads.x = threads.x;
        cycle_threads.y = threads.x;

        find_min_edges<<<blocks.x, threads.x>>>(vertices, edges, min_edges, curr_n_comp, n_vertices);
        cudaDeviceSynchronize();
        __syncthreads();
        remove_cycles<<<cycle_blocks, cycle_threads>>>(min_edges, curr_n_comp);
        cudaDeviceSynchronize();
        __syncthreads();
        merge<<<blocks.x, threads.x>>>(vertices, edges, min_edges, n_components, threads.y, blocks.y, n_vertices);
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
