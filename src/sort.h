//
// Created by gyorgy on 10/12/2020.
//

#ifndef FELZENSWALB_SORT_H
#define FELZENSWALB_SORT_H

#include "custom_types.h"

__device__ __forceinline__
int compare_min_edges(min_edge left, min_edge right) {
    //printf("Compare %d with %d\n", left.src_comp, right.src_comp);
    if (left.src_comp == 0 && right.src_comp == 0) return 0;
    if (left.src_comp == 0) return 1;
    if (right.src_comp == 0) return -1;
    uint component_diff = left.src_comp - right.src_comp;
    if (component_diff != 0) return component_diff;
    return left.weight - right.weight;
}

__global__
void sort_min_edges(min_edge min_edges[], uint vertices_length, uint offset, uint *not_sorted) {
    uint index = (blockDim.x * blockIdx.x + threadIdx.x) * 2 + offset;
    uint num_threads = gridDim.x * blockDim.x;

    for (uint tid = index; tid < vertices_length - 1; tid += (num_threads * 2)) {
        //printf("tid: %d\n", tid);
        min_edge left = min_edges[tid];
        min_edge right = min_edges[tid + 1];
        if (compare_min_edges(left, right) > 0) {
            //printf("Swap: %d with %d\n", tid, tid+1);
            min_edges[tid] = right;
            min_edges[tid+1] = left;
            *not_sorted = 1;
        }
    }
}

__device__ __forceinline__
void sort_min_edges(min_edge min_edges[], uint n_vertices, uint *did_change) {
    *did_change = 1;
    uint offset = 0;
    uint wanted_threads = n_vertices / 2;
    uint threads;
    uint blocks;
    if (wanted_threads <= 1024) {
        threads = wanted_threads;
        blocks = 1;
    } else {
        threads = 1024;
        blocks = wanted_threads / 1024 + 1;
    }
    while (*did_change == 1) {
        *did_change = 0;
        sort_min_edges<<<blocks, threads>>>(min_edges, n_vertices, offset, did_change);
        cudaDeviceSynchronize();
        offset ^= 1;
        sort_min_edges<<<blocks, threads>>>(min_edges, n_vertices, offset, did_change);
        cudaDeviceSynchronize();
        offset ^= 0;
    }
}

#endif //FELZENSWALB_SORT_H
