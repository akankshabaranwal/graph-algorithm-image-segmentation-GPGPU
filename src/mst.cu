//
// Created by gyorgy on 16/11/2020.
//

#include <cstdio>
#include <iostream>
#include <chrono>

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
        uint vertice_comp = vertices[tid].y;
        //vertices[tid].z = 1;
        uint min_weight = UINT_MAX;
        uint min_dest_comp = UINT_MAX;
        uint min_src_comp = 0;
        for (uint j = tid * NUM_NEIGHBOURS; j < tid * NUM_NEIGHBOURS + NUM_NEIGHBOURS; j++) {
            uint edge_id = edges[j].x;
            // Maybe it would be better to just check if it's not in the same component? We would not need to remove internal edges
            if (edge_id != 0) {
                uint edge_weight = edges[j].y;
                if (edge_weight <= min_weight) {
                    uint dest_comp = vertices[edge_id - 1].y;
                    if (edge_weight != min_weight || min_dest_comp > dest_comp) {
                        min_src_comp = vertice_comp;
                        min_dest_comp = dest_comp;
                        min_weight = edge_weight;
                    }
                }
            }
        }
        min_edges[tid].weight = min_weight;
        min_edges[tid].dest_comp = min_dest_comp;
        min_edges[tid].src_comp = min_src_comp;
    }
}

__global__
void reset_wrappers(min_edge_wrapper wrappers[], uint length) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;

    for (uint min_edge_id = tid; min_edge_id < length; min_edge_id += num_threads) {
        wrappers[min_edge_id].edge.y = UINT_MAX;
    }
}

__global__
void filter_min_edges_test(min_edge min_edges[], min_edge_wrapper new_min_edges[], uint length) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;

    for (uint min_edge_id = tid; min_edge_id < length; min_edge_id += num_threads) {
        min_edge our = min_edges[min_edge_id];
        if (our.src_comp < 1) continue;
        uint id = our.src_comp - 1;

        unsigned long long new_edge = our.weight;
        new_edge <<= 32;
        new_edge |= our.dest_comp;
        atomicMin_system((unsigned long long *)&new_min_edges[id].edge, new_edge);

        /*
        volatile uint2 *their_ptr = &(new_min_edges[id].edge);
        bool exit = false;
        while(!exit) {
            unsigned long long new_edge = our.weight;
            new_edge <<= 32;
            new_edge |= our.dest_comp;
            unsigned long long old = atomicMin_system((unsigned long long *)&new_min_edges[id].edge, new_edge);

            if (compare(our.dest_comp, our.weight, their_ptr->x, their_ptr->y) < 0) {
                unsigned long long new_edge = our.weight;
                new_edge <<= 32;
                new_edge |= our.dest_comp;
                printf("0x%X, 0x%X, 0x%llX\n", our.weight, our.dest_comp, new_edge);
                atomicExch_system((unsigned long long *)&new_min_edges[id].edge, new_edge);
                __threadfence_system();
            } else {
                exit = true;
            }
        }
        */
    }
}

__global__
void compact_min_edge_wrappers(min_edge min_edges[], min_edge_wrapper wrappers[], uint n_vertices, uint *pos_counter) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;
    for (int index = tid; index < n_vertices; index += num_threads) {
        uint2 edge = wrappers[index].edge;
        if (edge.y != UINT_MAX) {
            uint pos = atomicAdd_system(pos_counter, 1);
            min_edges[pos].src_comp = index + 1;
            min_edges[pos].dest_comp = edge.x;
            min_edges[pos].weight = edge.y;
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

        uint parent_comp = vertice->y;
        uint parent_id = v_id + 1;
        if (parent_id != parent_comp) {
            uint4 *parent;
            do {
                parent_id = parent_comp - 1;
                parent = &vertices[parent_id];
                parent_comp = parent->y;
                //printf("%d -> %d\n", parent_id + 1, parent_comp);
            } while(parent_id + 1 != parent_comp);

            vertice->y = parent_comp;
            atomicAdd_system(&(parent->z), vertice->z);
            //atomicAdd_system(&(parent->z), 1);
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
void merge(uint4 vertices[], min_edge min_edges[], uint *num_components, uint comp_count, uint k) {
    uint component_id = blockDim.x * blockIdx.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;
    for (uint comp_id = component_id; comp_id < comp_count; comp_id += num_threads) {

        min_edge min_edge = min_edges[comp_id];
        if (min_edge.src_comp == min_edge.dest_comp || min_edge.src_comp == 0) continue;
        uint4 src = vertices[min_edge.src_comp - 1];
        uint4 dest = vertices[min_edge.dest_comp - 1];
        uint src_diff = src.w + (k / src.z);
        uint dest_diff = dest.w + (k / dest.z);

        if (min_edge.weight <= min(src_diff, dest_diff)) {
            atomicSub_system(num_components, 1);
            uint new_int_diff = max(max(dest.w, src.w), min_edge.weight);
            min_edges[comp_id].weight = 0;
            min_edges[comp_id].new_int_diff = new_int_diff;
        }
    }
}

// Kernel to merge components based on size
__global__
void size_threshold(uint4 vertices[], min_edge min_edges[], uint *num_components, uint comp_count, uint min_size) {
    uint component_id = blockDim.x * blockIdx.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;
    for (uint comp_id = component_id; comp_id < comp_count; comp_id += num_threads) {

        min_edge min_edge = min_edges[comp_id];
        if (min_edge.src_comp == min_edge.dest_comp || min_edge.src_comp == 0) continue;

        uint size = vertices[min_edge.src_comp - 1].z;
        if (size <= min_size) {
            atomicSub_system(num_components, 1);
            min_edges[comp_id].weight = 0;
        }
    }
}

__global__
void debug_print_min_edges(min_edge min_edges[], uint length) {
    for (int i = 0; i < length; i++) {
        if (min_edges[i].src_comp == 0) continue;
        //printf("[%d]: %d(%d) -(%d)-> %d (%d)\n", i, min_edges[i].src_comp, min_edges[i].src_id, min_edges[i].weight, min_edges[i].dest_comp, min_edges[i].dest_id);
        //printf("[%d]: %d -(%d)-> %d\n", i, min_edges[i].src_comp, min_edges[i].weight, min_edges[i].dest_comp);
        printf("%d -(%d)-> %d\n", min_edges[i].src_comp, min_edges[i].weight, min_edges[i].dest_comp);
        //printf("%d -(%d)-> X\n", min_edges[i].src_comp, min_edges[i].weight);
    }
    printf("\n");
}

__device__
void debug_print_vertice(uint4 vertices[], uint pos, uint2 edges[]) {
    printf("vertices[%d] = %d %d %d %d | ", pos, vertices[pos].x, vertices[pos].y, vertices[pos].z, vertices[pos].w);
    for (uint j = pos * NUM_NEIGHBOURS; j < pos * NUM_NEIGHBOURS + NUM_NEIGHBOURS; j++) {
        printf("%d(%d), ", edges[j].x, edges[j].y);
    }
    printf("\n");
}

__global__
void debug_print_vertices(uint4 vertices[], uint length, uint2 edges[]) {
    for (int v_id = 0; v_id < length; v_id++) {
        debug_print_vertice(vertices, v_id, edges);
    }
}


__global__
void debug_print_comps(uint4 vertices[], uint length, uint2 edges[]) {
    for (int v_id = 0; v_id < length; v_id++) {
        if (vertices[v_id].x == vertices[v_id].y)
            debug_print_vertice(vertices, v_id, edges);
    }
}

__device__
void debug_print_comp(uint4 vertices[], uint length, uint comp_id, uint2 edges[]) {
    for (int v_id = 0; v_id < length; v_id++) {
        if (vertices[v_id].y == comp_id)
            debug_print_vertice(vertices, v_id, edges);
    }
}

// Kernel to orchestrate
__global__
void segment(uint4 vertices[], uint2 edges[], min_edge min_edges[], min_edge_wrapper wrappers[], uint2 sources[], uint *n_components, uint *did_change, uint k, uint min_size)
{
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
    while (curr_n_comp != prev_n_components && curr_n_comp > 1) {
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
            //printf("Filter\n");
            reset_wrappers<<<blocks.y, threads.y>>>(wrappers, n_vertices);
            filter_min_edges_test<<<blocks.y, threads.y>>>(min_edges, wrappers, n_vertices);

            //printf("Compact\n");
            *did_change = 0;
            compact_min_edge_wrappers<<<blocks.y, threads.y>>>(min_edges, wrappers, n_vertices, did_change);
        }

        //printf("Remove cycles\n");
        remove_deps(min_edges, curr_n_comp, sources ,blocks.x, threads.x, did_change);

        //printf("Merge\n");
        merge<<<blocks.x, threads.x>>>(vertices, min_edges, n_components, curr_n_comp, k);
        prev_n_components = curr_n_comp;
        cudaDeviceSynchronize();
        curr_n_comp = *n_components;
        if (prev_n_components == curr_n_comp) {
            size_threshold<<<blocks.x, threads.x>>>(vertices, min_edges, n_components, curr_n_comp, powf(min_size, counter));
            update_parents<<<blocks.x, threads.x>>>(vertices, min_edges, prev_n_components);
            path_compression<<<blocks.y, threads.y>>>(vertices, n_vertices);
            break;
        }

        //printf("Update\n");
        update_parents<<<blocks.x, threads.x>>>(vertices, min_edges, prev_n_components);

        //printf("Path compress\n");
        path_compression<<<blocks.y, threads.y>>>(vertices, n_vertices);

        //printf("New size\n");
        update_new_size<<<blocks.y, threads.y>>>(vertices, n_vertices, edges);

        //printf("N components: %d\n", curr_n_comp);
        counter++;
    }
    //printf("Iterations: %d\n", counter);
}

void remove_deps_cpu(min_edge min_edges[], uint num_components, uint2 sources[], uint blocks, uint threads, uint* did_change) {
    uint zero = 0;
    uint should_continue = 1;
    while (should_continue == 1) {
        cudaMemcpy(did_change, &(zero), sizeof(uint), cudaMemcpyHostToDevice);
        construct_sources<<<blocks, threads>>>(min_edges, num_components, sources);
        update_destinations<<<blocks, threads>>>(min_edges, num_components, sources, did_change);
        cudaDeviceSynchronize();
        cudaMemcpy(&should_continue, did_change, sizeof(uint), cudaMemcpyDeviceToHost);
    }
}

void segment_cpu(uint4 vertices[], uint2 edges[], min_edge min_edges[], min_edge_wrapper wrappers[], uint2 sources[], uint *n_components, uint *did_change, uint n_vertices, uint k, uint min_size) {
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
            //printf("Filter\n");
            reset_wrappers<<<blocks.y, threads.y>>>(wrappers, n_vertices);
            filter_min_edges_test<<<blocks.y, threads.y>>>(min_edges, wrappers, n_vertices);

            //printf("Compact\n");
            cudaMemcpy(did_change, &(zero), sizeof(uint), cudaMemcpyHostToDevice);
            compact_min_edge_wrappers<<<blocks.y, threads.y>>>(min_edges, wrappers, n_vertices, did_change);
        }

        //printf("Remove cycles\n");
        remove_deps_cpu(min_edges, curr_n_comp, sources ,blocks.x, threads.x, did_change);

        //printf("Merge\n");
        merge<<<blocks.x, threads.x>>>(vertices, min_edges, n_components, curr_n_comp, k);
        cudaDeviceSynchronize();
        prev_n_components = curr_n_comp;
        cudaMemcpy(&curr_n_comp, n_components, sizeof(uint), cudaMemcpyDeviceToHost);
        if (prev_n_components == curr_n_comp) {
            size_threshold<<<blocks.x, threads.x>>>(vertices, min_edges, n_components, curr_n_comp, static_cast<int>(powf(min_size, counter)));
            update_parents<<<blocks.x, threads.x>>>(vertices, min_edges, prev_n_components);
            path_compression<<<blocks.y, threads.y>>>(vertices, n_vertices);
            break;
        }

        //printf("Update\n");
        update_parents<<<blocks.x, threads.x>>>(vertices, min_edges, prev_n_components);

        //printf("Path compress\n");
        path_compression<<<blocks.y, threads.y>>>(vertices, n_vertices);
        //cudaDeviceSynchronize();

        //printf("New size\n");
        update_new_size<<<blocks.y, threads.y>>>(vertices, n_vertices, edges);
        //cudaDeviceSynchronize();

        //printf("N components: %d\n", curr_n_comp);
        counter++;
    }
    //printf("Iterations: %d\n", counter);
}

void checkErrors(const char *identifier) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << " " << identifier << std::endl;
        exit(1);
    }
}

char *compute_segments(void *input, uint x, uint y, size_t pitch, bool use_cpu, uint k, uint min_size) {
    uint num_vertices = (x) * (y);
    uint4 *vertices;
    uint2 *edges;

    cudaMalloc(&vertices, num_vertices*sizeof(uint4));
    checkErrors("Malloc vertices");
    cudaMalloc(&edges, num_vertices*NUM_NEIGHBOURS*sizeof(uint2));
    checkErrors("Malloc edges");

    // Write to the matrix from image
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

    uint2 *sources;
    min_edge *min_edges;
    min_edge_wrapper *wrappers;
    uint *num_components;
    uint *did_change;

    cudaMalloc(&num_components, sizeof(uint));
    checkErrors("Malloc num components");
    cudaMemcpyAsync(num_components, &num_vertices, sizeof(uint), cudaMemcpyHostToDevice);
    checkErrors("Memcpy num_vertices");

    cudaMalloc(&min_edges, num_vertices*sizeof(min_edge)); // max(min_edges) == vertices.length
    checkErrors("Malloc min_edges");
    cudaMalloc(&wrappers, num_vertices*sizeof(min_edge_wrapper)); // max(min_edges) == vertices.length
    checkErrors("Malloc min_edge wrappers");
    cudaMalloc(&sources, num_vertices*sizeof(uint2));
    checkErrors("Malloc sources");
    cudaMalloc(&did_change, sizeof(uint));
    checkErrors("Malloc did change");

    // Segment matrix
    if (!use_cpu) {
        segment<<<1, 1>>>(vertices, edges, min_edges, wrappers, sources, num_components, did_change, k, min_size);
    } else {
        segment_cpu(vertices, edges, min_edges, wrappers, sources, num_components, did_change, num_vertices, k, min_size);
    }

    // Generate image output
    dim3 decode_threads;
    dim3 decode_blocks;
    if (num_vertices <= 1024) {
        decode_threads.x = num_vertices;
        decode_blocks.x = 1;
    } else {
        decode_threads.x = 1024;
        decode_blocks.x = num_vertices / 1024 + 1;
    }

    char *output = (char*) malloc(x*y*CHANNEL_SIZE*sizeof(char));
    char *output_dev;
    cudaMalloc(&output_dev, num_vertices * CHANNEL_SIZE * sizeof(char));

    // Free everything we can
    cudaFree(edges);
    checkErrors("Free edges");
    cudaFree(min_edges);
    checkErrors("Free min_edges");
    cudaFree(wrappers);
    checkErrors("Free min_edge_wrappers");
    cudaFree(num_components);
    checkErrors("Free num_components");
    cudaFree(did_change);
    checkErrors("Free did_change");
    cudaFree(sources);
    checkErrors("Free sources");

    // Write image back from segmented matrix
    decode<<<decode_blocks, decode_threads>>>(vertices, output_dev, num_vertices);

    //Copy image data back from GPU
    cudaMemcpy(output, output_dev, x*y*CHANNEL_SIZE*sizeof(char), cudaMemcpyDeviceToHost);

    // Free rest
    cudaFree(vertices);
    checkErrors("Free vertices");

    cudaFree(output_dev);
    checkErrors("Free output_dev");

    return output;
}

char *compute_segments_partial(void *input, uint x, uint y, size_t pitch, bool use_cpu, uint k, uint min_size) {
    std::chrono::high_resolution_clock::time_point start, end;

    start = std::chrono::high_resolution_clock::now();
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

    // Write to the matrix from image
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

    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto time_span_encode = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << time_span_encode.count() << ",";
    checkErrors("encode()");

    // Segment matrix
    start = std::chrono::high_resolution_clock::now();
    if (!use_cpu) {
        segment<<<1, 1>>>(vertices, edges, min_edges, wrappers, sources, num_components, did_change, k, min_size);
    } else {
        segment_cpu(vertices, edges, min_edges, wrappers, sources, num_components, did_change, num_vertices, k, min_size);
    }
    cudaDeviceSynchronize();

    end = std::chrono::high_resolution_clock::now();
    auto time_span_segment = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << time_span_segment.count() << ",";

    checkErrors("segment()");

    // Free everything we can
    cudaFree(edges);
    checkErrors("Free edges");
    cudaFree(min_edges);
    checkErrors("Free min_edges");
    cudaFree(wrappers);
    checkErrors("Free min_edge_wrappers");
    cudaFree(num_components);
    checkErrors("Free num_components");
    cudaFree(did_change);
    checkErrors("Free did_change");
    cudaFree(sources);
    checkErrors("Free sources");

    start = std::chrono::high_resolution_clock::now();
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

    char *output_dev;
    cudaMalloc(&output_dev, num_vertices * CHANNEL_SIZE * sizeof(char));

    // Write image back from segmented matrix
    decode<<<decode_blocks, decode_threads>>>(vertices, output_dev, num_vertices);
    char *output = (char*) malloc(x*y*CHANNEL_SIZE*sizeof(char));
    cudaDeviceSynchronize();
    cudaMemcpy(output, output_dev, x*y*CHANNEL_SIZE*sizeof(char), cudaMemcpyDeviceToHost);

    end = std::chrono::high_resolution_clock::now();
    auto time_span_decode = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << time_span_decode.count() << std::endl;

    // Free rest
    cudaFree(vertices);
    checkErrors("Free vertices");

    //Copy image data back from GPU
    checkErrors("Memcpy output");

    cudaFree(output_dev);
    checkErrors("Free output_dev");

    return output;
}
