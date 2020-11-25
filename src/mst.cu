//
// Created by gyorgy on 16/11/2020.
//

#include <stdio.h>
#include <iostream>

#include "mst.h"

#define CHANNEL_SIZE 3
#define K 250
#define NUM_NEIGHBOURS 8

/*
 * Matrix structure:
 *      - vertices array of type uint4, where
 *          * x = id (starting at 1, so that 1 can represent null node)
 *          * y = component id
 *          * z = component size
 *          * w = component internal difference // This probably needs to be a float?
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
void encode(u_char *image, uint4 vertices[], uint3 edges[], uint x_len, uint y_len, size_t pitch) {
    uint x_pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (x_pos >= x_len) return;
    uint y_pos = blockDim.y * blockIdx.y + threadIdx.y;
    if (y_pos >= y_len) return;

    uint this_id = (x_pos * y_len + y_pos);
    uint4 *this_vertice = &vertices[this_id];
    this_vertice->x = this_id + 1;
    this_vertice->y = this_id + 1;
    this_vertice->z = 1;
    this_vertice->w = 0;

    uint this_start = x_pos * pitch + y_pos * CHANNEL_SIZE;
    u_char this_r = image[this_start];
    u_char this_g = image[this_start + 1];
    u_char this_b = image[this_start + 2];

    // Maybe could have 4 edges instead of 8?
    uint3 *edge;
    uint edge_id;
    uint other_start;
    u_char other_r;
    u_char other_g;
    u_char other_b;
    bool is_first_col = y_pos <= 0;
    bool is_last_col = y_pos >= y_len - 1;

    if (x_pos > 0) {
        uint prev_row = this_id - y_len;
        if (!is_first_col) {
            edge_id = prev_row - 1;
            other_start = (x_pos - 1) * pitch + (y_pos - 1) * CHANNEL_SIZE;
            other_r = image[other_start];
            other_g = image[other_start + 1];
            other_b = image[other_start + 2];
            edge = &edges[this_id * NUM_NEIGHBOURS];
            edge->x = edge_id + 1;
            edge->y = sqrtf(powf(this_r-other_r, 2.0f) + powf(this_g-other_g, 2.0f) + powf(this_b-other_b, 2.0f));
            edge->z = edge_id + 1;
        }

        edge_id = prev_row;
        other_start = (x_pos - 1) * pitch + (y_pos) * CHANNEL_SIZE;
        other_r = image[other_start];
        other_g = image[other_start + 1];
        other_b = image[other_start + 2];
        edge = &edges[this_id * NUM_NEIGHBOURS + 1];
        edge->x = edge_id + 1;
        edge->y = sqrtf(powf(this_r-other_r, 2.0f) + powf(this_g-other_g, 2.0f) + powf(this_b-other_b, 2.0f));
        edge->z = edge_id + 1;

        if (!is_last_col) {
            edge_id = prev_row + 1;
            other_start = (x_pos - 1) * pitch + (y_pos + 1) * CHANNEL_SIZE;
            other_r = image[other_start];
            other_g = image[other_start + 1];
            other_b = image[other_start + 2];
            edge = &edges[this_id * NUM_NEIGHBOURS + 2];
            edge->x = edge_id + 1;
            edge->y = sqrtf(powf(this_r-other_r, 2.0f) + powf(this_g-other_g, 2.0f) + powf(this_b-other_b, 2.0f));
            edge->z = edge_id + 1;
        }
    }

    if (x_pos < x_len - 1) {
        uint next_row = this_id + y_len;
        if (!is_first_col) {
            edge_id = next_row - 1;
            other_start = (x_pos + 1) * pitch + (y_pos - 1) * CHANNEL_SIZE;
            other_r = image[other_start];
            other_g = image[other_start + 1];
            other_b = image[other_start + 2];
            edge = &edges[this_id * NUM_NEIGHBOURS + 3];
            edge->x = edge_id + 1;
            edge->y = sqrtf(powf(this_r-other_r, 2.0f) + powf(this_g-other_g, 2.0f) + powf(this_b-other_b, 2.0f));
            edge->z = edge_id + 1;
        }

        edge_id = next_row;
        other_start = (x_pos + 1) * pitch + (y_pos) * CHANNEL_SIZE;
        other_r = image[other_start];
        other_g = image[other_start + 1];
        other_b = image[other_start + 2];
        edge = &edges[this_id * NUM_NEIGHBOURS + 4];
        edge->x = edge_id + 1;
        edge->y = sqrtf(powf(this_r-other_r, 2.0f) + powf(this_g-other_g, 2.0f) + powf(this_b-other_b, 2.0f));
        edge->z = edge_id + 1;

        if (!is_last_col) {
            edge_id = next_row + 1;
            other_start = (x_pos + 1) * pitch + (y_pos + 1) * CHANNEL_SIZE;
            other_r = image[other_start];
            other_g = image[other_start + 1];
            other_b = image[other_start + 2];
            edge = &edges[this_id * NUM_NEIGHBOURS + 5];
            edge->x = edge_id + 1;
            edge->y = sqrtf(powf(this_r-other_r, 2.0f) + powf(this_g-other_g, 2.0f) + powf(this_b-other_b, 2.0f));
            edge->z = edge_id + 1;
        }
    }

    if (!is_first_col) {
        edge_id = this_id - 1;
        other_start = (x_pos) * pitch + (y_pos - 1) * CHANNEL_SIZE;
        other_r = image[other_start];
        other_g = image[other_start + 1];
        other_b = image[other_start + 2];
        edge = &edges[this_id * NUM_NEIGHBOURS + 6];
        edge->x = edge_id + 1;
        edge->y = sqrtf(powf(this_r-other_r, 2.0f) + powf(this_g-other_g, 2.0f) + powf(this_b-other_b, 2.0f));
        edge->z = edge_id + 1;
    }

    if (!is_last_col) {
        edge_id = this_id + 1;
        other_start = (x_pos) * pitch + (y_pos + 1) * CHANNEL_SIZE;
        other_r = image[other_start];
        other_g = image[other_start + 1];
        other_b = image[other_start + 2];
        edge = &edges[this_id * NUM_NEIGHBOURS + 7];
        edge->x = edge_id + 1;
        edge->y = sqrtf(powf(this_r-other_r, 2.0f) + powf(this_g-other_g, 2.0f) + powf(this_b-other_b, 2.0f));
        edge->z = edge_id + 1;
    }
}

// Kernel to decode graph
__global__
void decode(uint4 vertices[], char *image, char* colours, uint num_vertices) {
    uint pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= num_vertices) return;

    uint img_pos = pos * CHANNEL_SIZE;
    uint colour_start = (vertices[pos].y - 1) * CHANNEL_SIZE;
    image[img_pos] = colours[colour_start];
    image[img_pos + 1] = colours[colour_start + 1];
    image[img_pos + 2] = colours[colour_start + 2];
}

// Kernel to find min edge
__global__
void find_min_edges(uint4 vertices[], uint3 edges[], uint3 min_edges[], uint num_components, uint vertices_length) {
    uint component_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (component_id >= num_components) return;

    uint3 min;
    min.x = UINT_MAX;
    min.y = 0;
    // Scan all vertices and find the min with component == tid
    for (int i = 0; i < vertices_length; i++) {
        uint4 vertice = vertices[i];
        if (vertice.y - 1 == component_id) {
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
void remove_cycles(uint3 min_edges[], uint num_components, uint *did_change) {
    uint component_id_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (component_id_x >= num_components) return;

    uint component_id_y = blockDim.y * blockIdx.y + threadIdx.y;
    if (component_id_y >= num_components) return;

    if (component_id_x == component_id_y) return;

    uint3 *edge = &min_edges[component_id_x];
    uint src = edge->y;
    uint dest = edge->z;
    if (src == dest) return;
    __syncthreads();

    uint3 *curr_edge = &min_edges[component_id_y];
    if (src == curr_edge->z) {
        if (component_id_x > component_id_y || dest != curr_edge->y) {
            curr_edge->z = dest;
            *did_change = 1;
        }
    }
}

// Kernel to update vertices with new components
__global__
void update_matrix(uint4 vertices[], uint3 edges[], uint vertices_length, uint new_component, uint new_size, uint new_int_diff, uint original_id) {
    uint vertice_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (vertice_id >= vertices_length) return;

    uint4 *vertice = &vertices[vertice_id];
    bool is_vertice_new_comp = vertice->y == original_id || vertice->y == new_component;
    if (is_vertice_new_comp) {
        vertice->y = new_component;
        vertice->z = new_size;
        vertice->w = new_int_diff;
    }

    for (int j = vertice_id * NUM_NEIGHBOURS; j < vertice_id * NUM_NEIGHBOURS + NUM_NEIGHBOURS; j++) {
        uint3 *neighbour_edge = &edges[j];
        if (neighbour_edge->x != 0) {
            if (neighbour_edge->z == original_id || neighbour_edge->z == new_component) {
                if (is_vertice_new_comp) neighbour_edge->x = 0; // Remove internal edges
                else neighbour_edge->z = new_component;
            }
        }
    }

}

// Kernel to merge components
__global__
void merge(uint4 vertices[], uint3 edges[], uint3 min_edges[], uint *num_components, uint update_threads, uint update_blocks, uint vertices_length, uint comp_count) {
    uint component_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (component_id >= comp_count)  {
        return;
    }

    uint3 min_edge = min_edges[component_id];
    if (min_edge.y == min_edge.z || min_edge.y == 0) return;
    uint4 src = vertices[min_edge.y - 1];
    uint4 dest = vertices[min_edge.z - 1];
    __syncthreads();
    uint src_diff = src.w + (K / src.z);
    uint dest_diff = dest.w + (K / dest.z);
    if (min_edge.x <= min(src_diff, dest_diff)) {
        atomicSub_system(num_components, 1); // Is this horribly inefficient?
        uint new_int_diff = max(max(dest.w, src.w), min_edge.x);
        uint new_size = src.z + dest.z;
        uint new_component = dest.y;

        update_matrix<<<update_blocks, update_threads>>>(vertices, edges, vertices_length, new_component, new_size, new_int_diff, src.y);
        cudaDeviceSynchronize();
    }
}

__global__
void debug_print_min_edges(uint3 min_edges[], uint length) {
    for (int i = 0; i < length; i++) {
        printf("(%d %d)\n", min_edges[i].y, min_edges[i].z);
    }
    printf("\n");
}

// Kernel to orchestrate
__global__
void segment(uint4 vertices[], uint3 edges[], uint3 min_edges[], uint *n_components, uint *did_change) {
    uint prev_n_components = 0;
    uint n_vertices = *n_components;
    uint curr_n_comp = *n_components;
    dim3 threads;
    dim3 blocks;
    dim3 cycle_blocks;
    dim3 cycle_threads;
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

        if (curr_n_comp < 32) {
            cycle_threads.x = curr_n_comp;
            cycle_threads.y = cycle_threads.x;

            cycle_blocks.x = 1;
            cycle_blocks.y = 1;
        } else {
            cycle_threads.x = 32;
            cycle_blocks.x = curr_n_comp / 32 + 1;

            cycle_threads.y = cycle_threads.x;
            cycle_blocks.y = cycle_blocks.x;
        }

        find_min_edges<<<blocks.x, threads.x>>>(vertices, edges, min_edges, curr_n_comp, n_vertices);
        cudaDeviceSynchronize();

        *did_change = 1;
        while (*did_change == 1) {
            *did_change = 0;
            remove_cycles<<<cycle_blocks, cycle_threads>>>(min_edges, curr_n_comp, did_change);
            cudaDeviceSynchronize();
        }

        merge<<<blocks.x, threads.x>>>(vertices, edges, min_edges, n_components, threads.y, blocks.y, n_vertices, curr_n_comp);
        cudaDeviceSynchronize();

        prev_n_components = curr_n_comp;
        curr_n_comp = *n_components;
        printf("N components: %d\n", curr_n_comp);
    }
}

__global__
void debug_print_vertices(uint4 vertices[], uint length, uint3 edges[]) {
    for (int v_id = 0; v_id < length; v_id++) {
        printf("vertices[%d] = %d %d | ", v_id, vertices[v_id].x, vertices[v_id].y);
        for (int j = v_id * NUM_NEIGHBOURS; j < v_id * NUM_NEIGHBOURS + NUM_NEIGHBOURS; j++) {
            printf("%d(%d), ", edges[j].x, edges[j].y);
        }
        printf("\n");
    }
}

void get_component_colours(char colours[], uint num_colours) {
    srand(123456789);
    for (int i = 0; i < num_colours * CHANNEL_SIZE; i++) {
        colours[i] = rand() % 256;
    }
}

void checkErrors(const char *identifier) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << " " << identifier << std::endl;
}

char *compute_segments(void *input, uint x, uint y, size_t pitch) {
    uint4 *vertices;
    uint3 *edges;
    uint3 *min_edges;
    uint num_vertices = (x) * (y);
    uint *num_components;
    uint *did_change;

    cudaMalloc(&vertices, num_vertices*sizeof(uint4));
    checkErrors("Malloc vertices");
    cudaMalloc(&edges, num_vertices*NUM_NEIGHBOURS*sizeof(uint3));
    checkErrors("Malloc edges");
    cudaMalloc(&min_edges, num_vertices*sizeof(uint3)); // max(min_edges) == vertices.length
    checkErrors("Malloc min_edges");
    cudaMalloc(&num_components, sizeof(uint));
    checkErrors("Malloc num components");
    cudaMalloc(&did_change, sizeof(uint));
    checkErrors("Malloc did change");

    cudaMemcpyAsync(num_components, &num_vertices, sizeof(uint), cudaMemcpyHostToDevice);
    checkErrors("Memcpy num_vertices");

    // Write to the matrix from image
    // cudaOccupancyScheduler?
    dim3 encode_threads;
    dim3 encode_blocks;
    if (num_vertices < 1024) {
        encode_threads.x = x;
        encode_threads.y = y;
        encode_blocks.x = 1;
        encode_blocks.y = 1;
    } else {
        encode_threads.x = 32;
        encode_threads.y = 32;
        encode_blocks.x = x / 32 + 1;
        encode_blocks.y = y / 32 + 1;
    }

    encode<<<encode_blocks, encode_threads>>>((u_char*)input, vertices, edges, x, y, pitch);
    checkErrors("encode()");

    // Segment matrix
    segment<<<1, 1>>>(vertices, edges, min_edges, num_components, did_change);
    cudaDeviceSynchronize();
    checkErrors("segment()");

    // Setup random colours for components
    dim3 decode_threads;
    dim3 decode_blocks;
    if (num_vertices <= 1024) {
        decode_threads.x = num_vertices;
        decode_blocks.x = 1;
    } else {
        decode_threads.x = 1024;
        decode_blocks.x = num_vertices / 1024 + 1;
    }

    char component_colours[num_vertices * CHANNEL_SIZE];
    get_component_colours(component_colours, num_vertices);
    char *component_colours_dev;
    cudaMalloc(&component_colours_dev, num_vertices * CHANNEL_SIZE * sizeof(char));
    cudaMemcpyAsync(component_colours_dev, component_colours, num_vertices * CHANNEL_SIZE * sizeof(char), cudaMemcpyHostToDevice);
    char *output_dev;
    cudaMalloc(&output_dev, num_vertices * CHANNEL_SIZE * sizeof(char ));

    // Write image back from segmented matrix
    decode<<<decode_blocks, decode_threads>>>(vertices, output_dev, component_colours_dev, num_vertices);
    cudaDeviceSynchronize();
    checkErrors("decode()");

    // Clean up matrix
    cudaFree(vertices);
    checkErrors("Free vertices");
    cudaFree(edges);
    checkErrors("Free edges");
    cudaFree(min_edges);
    checkErrors("Free min_edges");
    cudaFree(component_colours_dev);
    checkErrors("Free component_colours_dev");

    //Copy image data back from GPU
    char *output = (char*) malloc(x*y*CHANNEL_SIZE*sizeof(char));

    cudaMemcpy(output, output_dev, x*y*CHANNEL_SIZE*sizeof(char), cudaMemcpyDeviceToHost);
    checkErrors("Memcpy output");

    return output;
}
