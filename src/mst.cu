//
// Created by gyorgy on 16/11/2020.
//

#include <stdio.h>
#include <iostream>

#include "mst.h"
#include "sort.h"
#include "custom_types.h"
#include "graph.h"

/*
 * Matrix structure:
 *      - vertices array of type uint4, where
 *          * x = id (starting at 1, so that 1 can represent null node)
 *          * y = component id
 *          * z = component size
 *          * w = component internal difference // This probably needs to be a float?
 *
 *      - edges 2D array of type uint2, where
 *          * x = destination id
 *          * y = weight (dissimilarity)
 *
 * Min edges:
 *      - x = weight
 *      - y = source id
 *      - z = destination id
 *      - w = component id
 */

__global__
void find_min_edges_sort(uint4 vertices[], uint2 edges[], min_edge min_edges[], uint vertices_length) {
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;
    for (uint tid = index; tid < vertices_length; tid += num_threads) {
        uint4 vertice = vertices[tid];
        min_edge min;
        min.weight = UINT_MAX;
        min.src_comp = 0;
        for (int j = tid * NUM_NEIGHBOURS; j < tid * NUM_NEIGHBOURS + NUM_NEIGHBOURS; j++) {
            uint2 edge = edges[j];
            // Maybe it would be better to just check if it's not in the same component? We would not need to remove internal edges
            if (edge.x != 0) {
                if (edge.y < min.weight) {
                    min.src_comp = vertice.y;
                    min.dest_comp = vertices[edge.x - 1].y; //edge.z;
                    min.weight = edge.y;
                }
            }
        }
        min_edges[tid] = min;
    }
}

__global__
void reset_wrappers(min_edge_wrapper wrappers[], uint length) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;

    for (uint min_edge_id = tid; min_edge_id < length; min_edge_id += num_threads) {
        wrappers[min_edge_id].edge.src_comp = 0;
        wrappers[min_edge_id].locked = 0;
    }
}

__global__
void filter_min_edges(min_edge min_edges[], min_edge_wrapper new_min_edges[], uint length) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;

    for (uint min_edge_id = tid; min_edge_id < length; min_edge_id += num_threads) {
        min_edge our = min_edges[min_edge_id];
        uint id = our.src_comp;

        uint lock;
        min_edge their;
        bool exit = false;
        while(!exit) {
            their = new_min_edges[id].edge;
            if (compare_min_edges(our, their) < 0) {
                lock = atomicCAS_system(&(new_min_edges[id].locked), 0, 1);

                if (lock == 0) {
                    min_edge updated_their = new_min_edges[id].edge;
                    if (compare_min_edges(our, updated_their) < 0) {
                        new_min_edges[id].edge = our;
                        __threadfence_system();
                    }
                    atomicExch_system(&(new_min_edges[id].locked), 0);
                    exit = true;
                }
                __threadfence_system();
            } else {
                exit = true;
            }
        }
    }
}

__global__
void compact_min_edge_wrappers(min_edge min_edges[], min_edge_wrapper wrappers[], uint n_vertices, uint *pos_counter) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;
    for (int index = tid; index < n_vertices; index += num_threads) {
        min_edge edge = wrappers[index].edge;
        if (edge.src_comp != 0) {
            uint pos = atomicAdd_system(pos_counter, 1);
            min_edges[pos] = edge;
        }
    }
}

__global__
void construct_sources(min_edge min_edges[], uint num_components, uint2 sources[]) {
    uint component_id = blockDim.x * blockIdx.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;
    for (uint comp_id = component_id; comp_id < num_components; comp_id += num_threads) {
        min_edge *edge = &min_edges[comp_id];
        sources[edge->src_comp - 1].x = edge->dest_comp;
        sources[edge->src_comp - 1].y = edge->weight;
    }
}

__global__
void update_destinations(min_edge min_edges[], uint num_components, uint2 sources[], uint *did_change) {
    uint component_id = blockDim.x * blockIdx.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;
    for (uint comp_id = component_id; comp_id < num_components; comp_id += num_threads) {
        min_edge *edge = &min_edges[comp_id];
        uint src = edge->src_comp;
        uint dest = edge->dest_comp;
        uint weight = edge->weight;
        uint new_dest = sources[dest - 1].x;
        uint new_weight = sources[dest - 1].y;
        if (((new_dest == src) || (new_dest != src && new_dest != dest && weight == new_weight)) && src < dest) {
            edge->dest_comp = new_dest;
            *did_change = 1;
        }
    }
}

__device__ __forceinline__
void remove_deps(min_edge min_edges[], uint num_components, uint2 sources[], uint blocks, uint threads, uint* did_change) {
    *did_change = 1;
    while (*did_change == 1) {
        *did_change = 0;
        construct_sources<<<blocks, threads>>>(min_edges, num_components, sources);
        update_destinations<<<blocks, threads>>>(min_edges, num_components, sources, did_change);
        cudaDeviceSynchronize();
    }
}

// Kernel to update the whole matrix
__global__
void update_parents(uint4 vertices[], min_edge min_edges[], uint num_components) {
    uint component_id = blockDim.x * blockIdx.x + threadIdx.x;
    uint comp_threads = gridDim.x * blockDim.x;

    for (int comp_id = component_id; comp_id < num_components; comp_id += comp_threads) {
        min_edge current_comp = min_edges[comp_id];
        if (current_comp.src_comp == current_comp.dest_comp || current_comp.weight > 0) continue;
        //printf("Merge %d into %d\n", current_comp.src_comp, current_comp.dest_comp);

        // If we merge
        vertices[current_comp.src_comp - 1].y = current_comp.dest_comp;
        atomicMax_system(&(vertices[current_comp.dest_comp - 1].w), current_comp.new_int_diff);
    }
}

__device__
void print_vertice(uint4 vertices[], uint pos) {
    uint component = vertices[pos].y;
    uint4 parent = vertices[component - 1];
    printf("[%d] (%d, %d) -> (%d, %d)\n", pos, vertices[pos].x, component, parent.x, parent.y);
}

__global__
void path_compression(uint4 vertices[], uint num_vertices) {
    uint vertice_id = blockDim.x * blockIdx.x + threadIdx.x;
    uint comp_threads = gridDim.x * blockDim.x;

    for (int v_id = vertice_id; v_id < num_vertices; v_id += comp_threads) {
        uint4 *vertice = &vertices[v_id];

        if (vertice->x != vertice->y) {
            uint4 *parent = &vertices[vertice->y - 1];
            while(parent->y != parent->x) {parent = &vertices[parent->y - 1]; if (false) printf("Parent %d -> %d\n", parent->x, parent->y);}

            vertice->y = parent->x;
            atomicAdd_system(&(parent->z), vertice->z);
            atomicMax_system(&(parent->w), vertice->w);
        }
    }
}

__global__
void update_new_size(uint4 vertices[], uint num_vertices, uint2 edges[]) {
    uint vertice_id = blockDim.x * blockIdx.x + threadIdx.x;
    uint comp_threads = gridDim.x * blockDim.x;

    for (int v_id = vertice_id; v_id < num_vertices; v_id += comp_threads) {
        uint4 *vertice = &vertices[v_id];

        if (vertice->x != vertice->y) {
            uint4 *parent = &vertices[vertice->y - 1];
            vertice->z = parent->z;
            vertice->w = parent->w;
        }

        for (int j = v_id * NUM_NEIGHBOURS; j < v_id * NUM_NEIGHBOURS + NUM_NEIGHBOURS; j++) {
            uint2 *neighbour_edge = &edges[j];
            if (neighbour_edge->x != 0) {
                if (vertices[neighbour_edge->x - 1].y == vertice->y) {
                    neighbour_edge->x = 0; // Remove internal edges
                }
            }
        }
    }
}

// Kernel to merge components
__global__
void merge(uint4 vertices[], min_edge min_edges[], uint *num_components, uint update_threads, uint update_blocks, uint vertices_length, uint comp_count) {
    uint component_id = blockDim.x * blockIdx.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;
    for (uint comp_id = component_id; comp_id < comp_count; comp_id += num_threads) {

        min_edge min_edge = min_edges[comp_id];
        if (min_edge.src_comp == min_edge.dest_comp || min_edge.src_comp == 0) return;
        uint4 src = vertices[min_edge.src_comp - 1];
        uint4 dest = vertices[min_edge.dest_comp - 1];
        uint src_diff = src.w + (K / src.z);
        uint dest_diff = dest.w + (K / dest.z);
        __syncthreads();

        if (min_edge.weight <= min(src_diff, dest_diff)) {
            atomicSub_system(num_components, 1);
            uint new_int_diff = max(max(dest.w, src.w), min_edge.weight);
            min_edges[comp_id].weight = 0;
            min_edges[comp_id].new_int_diff = new_int_diff;
        }
    }
}

__global__
void debug_print_min_edges(min_edge min_edges[], uint length) {
    for (int i = 0; i < length; i++) {
        if (min_edges[i].src_comp == 0) continue;
        //printf("[%d]: %d(%d) -(%d)-> %d (%d)\n", i, min_edges[i].src_comp, min_edges[i].src_id, min_edges[i].weight, min_edges[i].dest_comp, min_edges[i].dest_id);
        printf("[%d]: %d -(%d)-> %d\n", i, min_edges[i].src_comp, min_edges[i].weight, min_edges[i].dest_comp);
        //printf("%d -(%d)-> %d\n", min_edges[i].src_comp, min_edges[i].weight, min_edges[i].dest_comp);
    }
    printf("\n");
}

__global__
void debug_print_vertices(uint4 vertices[], uint length, uint2 edges[]) {
    for (int v_id = 0; v_id < length; v_id++) {
        printf("vertices[%d] = %d %d %d | ", v_id, vertices[v_id].x, vertices[v_id].y, vertices[v_id].z);
        for (int j = v_id * NUM_NEIGHBOURS; j < v_id * NUM_NEIGHBOURS + NUM_NEIGHBOURS; j++) {
            printf("%d(%d), ", edges[j].x, edges[j].y);
        }
        printf("\n");
    }
}

// Kernel to orchestrate
__global__
void segment(uint4 vertices[], uint2 edges[], min_edge min_edges[], min_edge_wrapper wrappers[], uint2 sources[], uint *n_components, uint *did_change) {
    uint counter = 0;
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
        blocks.y = min(n_vertices / 1024 + 1, 65535);
    }

    //printf("N components: %d\n", curr_n_comp);
    while (curr_n_comp != prev_n_components) {
        if (curr_n_comp < 1024) {
            threads.x = curr_n_comp;
            blocks.x = 1;
        } else {
            threads.x = 1024;
            blocks.x = min(curr_n_comp / 1024 + 1, 65535);
        }

        //printf("Find min edges\n");
        find_min_edges_sort<<<blocks.y, threads.y>>>(vertices, edges, min_edges, n_vertices);

        // First time there is no point in doing these, since n_vertices == n_components
        if (counter > 0) {
            //printf("Sort\n");
            reset_wrappers<<<blocks.y, threads.y>>>(wrappers, n_vertices);
            filter_min_edges<<<blocks.y, threads.y>>>(min_edges, wrappers, n_vertices);
            cudaDeviceSynchronize();

            //printf("Compact\n");
            *did_change = 0;
            compact_min_edge_wrappers<<<blocks.y, threads.y>>>(min_edges, wrappers, n_vertices, did_change);
        }

        //printf("Remove cycles\n");
        remove_deps(min_edges, curr_n_comp, sources ,blocks.x, threads.x, did_change);

        //printf("Merge\n");
        merge<<<blocks.x, threads.x>>>(vertices, min_edges, n_components, threads.y, blocks.y, n_vertices, curr_n_comp);

        //printf("Update\n");
        update_parents<<<blocks.x, threads.x>>>(vertices, min_edges, curr_n_comp);

        //printf("Path compress\n");
        path_compression<<<blocks.y, threads.y>>>(vertices, n_vertices);

        //printf("New size\n");
        update_new_size<<<blocks.y, threads.y>>>(vertices, n_vertices, edges);
        cudaDeviceSynchronize();

        prev_n_components = curr_n_comp;
        curr_n_comp = *n_components;
        //printf("N components: %d\n", curr_n_comp);
        counter++;
        //return;
    }
    //printf("Iterations: %d\n", counter);
}

void remove_deps_cpu(min_edge min_edges[], uint num_components, uint2 sources[], uint blocks, uint threads, uint* did_change) {
    //*did_change = 1;
    uint zero = 0;
    uint one = 1;
    uint should_continue = 1;
    cudaMemcpy(did_change, &(one), sizeof(uint), cudaMemcpyHostToDevice);
    while (should_continue == 1) {
        //*did_change = 0;
        cudaMemcpy(did_change, &(zero), sizeof(uint), cudaMemcpyHostToDevice);
        construct_sources<<<blocks, threads>>>(min_edges, num_components, sources);
        update_destinations<<<blocks, threads>>>(min_edges, num_components, sources, did_change);
        cudaDeviceSynchronize();
        cudaMemcpy(&should_continue, did_change, sizeof(uint), cudaMemcpyDeviceToHost);
    }
}

void segment_cpu(uint4 vertices[], uint2 edges[], min_edge min_edges[], min_edge_wrapper wrappers[], uint2 sources[], uint *n_components, uint *did_change, uint n_vertices) {
    uint zero = 0;
    uint counter = 0;
    uint prev_n_components = 0;
    uint curr_n_comp = n_vertices;
    dim3 threads;
    dim3 blocks;
    if (n_vertices < 1024) {
        threads.y = n_vertices;
        blocks.y = 1;
    } else {
        threads.y = 1024;
        blocks.y = std::min(n_vertices / 1024 + 1, (uint)65535);
    }

    //printf("N components: %d\n", curr_n_comp);
    while (curr_n_comp != prev_n_components) {
        if (curr_n_comp < 1024) {
            threads.x = curr_n_comp;
            blocks.x = 1;
        } else {
            threads.x = 1024;
            blocks.x = std::min(curr_n_comp / 1024 + 1, (uint)65535);
        }

        //printf("Find min edges\n");
        find_min_edges_sort<<<blocks.y, threads.y>>>(vertices, edges, min_edges, n_vertices);

        // First time there is no point in doing these, since n_vertices == n_components
        if (counter > 0) {
            //printf("Sort\n");
            reset_wrappers<<<blocks.y, threads.y>>>(wrappers, n_vertices);
            filter_min_edges<<<blocks.y, threads.y>>>(min_edges, wrappers, n_vertices);
            cudaDeviceSynchronize();

            //printf("Compact\n");
            //*did_change = 0;
            cudaMemcpy(did_change, &(zero), sizeof(uint), cudaMemcpyHostToDevice);
            compact_min_edge_wrappers<<<blocks.y, threads.y>>>(min_edges, wrappers, n_vertices, did_change);
        }

        //printf("Remove cycles\n");
        remove_deps_cpu(min_edges, curr_n_comp, sources ,blocks.x, threads.x, did_change);

        //printf("Merge\n");
        merge<<<blocks.x, threads.x>>>(vertices, min_edges, n_components, threads.y, blocks.y, n_vertices, curr_n_comp);

        //printf("Update\n");
        update_parents<<<blocks.x, threads.x>>>(vertices, min_edges, curr_n_comp);

        //printf("Path compress\n");
        path_compression<<<blocks.y, threads.y>>>(vertices, n_vertices);

        //printf("New size\n");
        update_new_size<<<blocks.y, threads.y>>>(vertices, n_vertices, edges);
        cudaDeviceSynchronize();

        prev_n_components = curr_n_comp;
        //curr_n_comp = *n_components;
        cudaMemcpy(&curr_n_comp, n_components, sizeof(uint), cudaMemcpyDeviceToHost);
        //printf("N components: %d\n", curr_n_comp);
        counter++;
        //return;
    }
    //printf("Iterations: %d\n", counter);
}

void get_component_colours(char colours[], uint num_colours) {
    srand(123456789);
    for (int i = 0; i < num_colours * CHANNEL_SIZE; i++) {
        colours[i] = rand() % 256;
    }
}

void checkErrors(const char *identifier) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << " " << identifier << std::endl;
        exit(1);
    }
}

char *compute_segments(void *input, uint x, uint y, size_t pitch, bool use_cpu) {
    uint4 *vertices;
    uint2 *edges;
    uint2 *sources;
    min_edge *min_edges;
    min_edge_wrapper *wrappers;
    uint num_vertices = (x) * (y);
    uint *num_components;
    uint *did_change;

    cudaMalloc(&vertices, num_vertices*sizeof(uint4));
    checkErrors("Malloc vertices");
    cudaMalloc(&edges, num_vertices*NUM_NEIGHBOURS*sizeof(uint2));
    checkErrors("Malloc edges");
    cudaMalloc(&min_edges, num_vertices*sizeof(min_edge)); // max(min_edges) == vertices.length
    checkErrors("Malloc min_edges");
    cudaMalloc(&wrappers, num_vertices*sizeof(min_edge_wrapper)); // max(min_edges) == vertices.length
    checkErrors("Malloc min_edge wrappers");
    cudaMalloc(&sources, num_vertices*sizeof(uint2));
    checkErrors("Malloc sources");
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
    //cudaSetDeviceFlags(cudaDeviceBlockingSync);
    if (!use_cpu) {
        segment<<<1, 1>>>(vertices, edges, min_edges, wrappers, sources, num_components, did_change);
        cudaDeviceSynchronize();
    } else {
        segment_cpu(vertices, edges, min_edges, wrappers, sources, num_components, did_change, num_vertices);
    }
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

    //char component_colours[num_vertices * CHANNEL_SIZE];
    char *component_colours = (char *) malloc(num_vertices * CHANNEL_SIZE * sizeof(char));
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
