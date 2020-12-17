//
// Created by gyorgy on 10/12/2020.
//

#ifndef FELZENSWALB_SORT_H
#define FELZENSWALB_SORT_H

#include "custom_types.h"

__device__ __forceinline__
int compare_min_edges_volatile(min_edge *left, min_edge volatile *right) {
    bool is_left_comp_zero = left->src_comp == 0;
    bool is_right_comp_zero = right->src_comp == 0;

    if (is_left_comp_zero && is_right_comp_zero) return 0;
    if (is_right_comp_zero) return -1;
    if (is_left_comp_zero) return 1;

    return left->weight - right->weight;
}

__device__ __forceinline__
int compare(uint left_weight, uint right_comp, uint right_weight) {
    bool is_right_comp_zero = right_comp == 0;

    if (is_right_comp_zero) return -1;

    return left_weight - right_weight;
}

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

__device__ __forceinline__
int compare_min_edges_with_pos(min_edge min_edges[], uint left_pos, uint right_pos, uint length, min_edge *left, min_edge *right) {
    bool is_left_ob = left_pos >= length;
    bool is_right_ob = right_pos >= length;
    if (is_left_ob && is_right_ob) return 0;
    if (is_left_ob) return 1;
    if (is_right_ob) return -1;

    *left = min_edges[left_pos];
    *right = min_edges[right_pos];

    return compare_min_edges(*left, *right);
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

__global__
void bitonic_step(min_edge min_edges[], uint step, uint direction, uint length, bool swap_direction) {
    uint thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;

    for (uint tid = thread_id; tid < length; tid += num_threads) {
        uint pair = tid ^ step;

        if (pair > tid) {
            min_edge left;
            min_edge right;
            uint dir = tid & direction;
            bool is_asc = swap_direction ? dir != 0 : dir == 0;
            bool is_desc = !is_asc;


            // Ascending
            if (is_asc) {
                if (compare_min_edges_with_pos(min_edges, tid, pair, length, &left, &right) > 0) {
                    if (pair < length && tid < length) {
                        min_edges[tid] = right;
                        min_edges[pair] = left;
                    }
                }
            }

            //Descending
            if (is_desc) {
                if (compare_min_edges_with_pos(min_edges, tid, pair, length, &left, &right) < 0) {
                    min_edges[tid] = right;
                    min_edges[pair] = left;
                }
            }
        }
    }
}

__device__ __forceinline__
void sort_min_edges_bitonic(min_edge min_edges[], uint n_vertices) {
    uint log_val = log2f(n_vertices);
    uint padded_size = (n_vertices != (uint)powf(2, log_val)) ? 1 << (log_val + 1): n_vertices;

    uint blocks = 1;
    uint threads = 1024;
    if (padded_size < 1024) threads = padded_size;
    else blocks = min(padded_size / 1024 + 1, 65535);

    /* Major step */
    bool swap_direction;
    for (uint k = 2; k <= padded_size; k <<= 1) {
        /* Minor step */
        swap_direction = (k > padded_size - n_vertices) && (k < padded_size);
        for (uint j = k >> 1; j > 0; j = j >> 1) {
            bitonic_step<<<blocks, threads>>>(min_edges, j, k, n_vertices, swap_direction);
            cudaDeviceSynchronize();
        }
    }
}

#endif //FELZENSWALB_SORT_H
