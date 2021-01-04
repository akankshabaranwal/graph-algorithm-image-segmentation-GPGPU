//
// Created by gyorgy on 10/12/2020.
//

#ifndef FELZENSWALB_GRAPH_H
#define FELZENSWALB_GRAPH_H

#include "custom_types.h"

__device__ __forceinline__
uint get_edge_weight(u_char this_r, u_char this_g, u_char this_b, u_char other_r, u_char other_g, u_char other_b) {
    return sqrtf(powf(this_r-other_r, 2.0f) + powf(this_g-other_g, 2.0f) + powf(this_b-other_b, 2.0f));
}

// Kernel to encode graph
__global__
void encode(u_char *image, uint4 vertices[], uint2 edges[], uint x_len, uint y_len, size_t pitch) {
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
    uint2 *edge;
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
            edge->y = get_edge_weight(this_r, this_g, this_b, other_r, other_g, other_b);
        }

        edge_id = prev_row;
        other_start = (x_pos - 1) * pitch + (y_pos) * CHANNEL_SIZE;
        other_r = image[other_start];
        other_g = image[other_start + 1];
        other_b = image[other_start + 2];
        edge = &edges[this_id * NUM_NEIGHBOURS + 1];
        edge->x = edge_id + 1;
        edge->y = get_edge_weight(this_r, this_g, this_b, other_r, other_g, other_b);

        if (!is_last_col) {
            edge_id = prev_row + 1;
            other_start = (x_pos - 1) * pitch + (y_pos + 1) * CHANNEL_SIZE;
            other_r = image[other_start];
            other_g = image[other_start + 1];
            other_b = image[other_start + 2];
            edge = &edges[this_id * NUM_NEIGHBOURS + 2];
            edge->x = edge_id + 1;
            edge->y = get_edge_weight(this_r, this_g, this_b, other_r, other_g, other_b);
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
            edge->y = get_edge_weight(this_r, this_g, this_b, other_r, other_g, other_b);
        }

        edge_id = next_row;
        other_start = (x_pos + 1) * pitch + (y_pos) * CHANNEL_SIZE;
        other_r = image[other_start];
        other_g = image[other_start + 1];
        other_b = image[other_start + 2];
        edge = &edges[this_id * NUM_NEIGHBOURS + 4];
        edge->x = edge_id + 1;
        edge->y = get_edge_weight(this_r, this_g, this_b, other_r, other_g, other_b);

        if (!is_last_col) {
            edge_id = next_row + 1;
            other_start = (x_pos + 1) * pitch + (y_pos + 1) * CHANNEL_SIZE;
            other_r = image[other_start];
            other_g = image[other_start + 1];
            other_b = image[other_start + 2];
            edge = &edges[this_id * NUM_NEIGHBOURS + 5];
            edge->x = edge_id + 1;
            edge->y = get_edge_weight(this_r, this_g, this_b, other_r, other_g, other_b);
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
        edge->y = get_edge_weight(this_r, this_g, this_b, other_r, other_g, other_b);
    }

    if (!is_last_col) {
        edge_id = this_id + 1;
        other_start = (x_pos) * pitch + (y_pos + 1) * CHANNEL_SIZE;
        other_r = image[other_start];
        other_g = image[other_start + 1];
        other_b = image[other_start + 2];
        edge = &edges[this_id * NUM_NEIGHBOURS + 7];
        edge->x = edge_id + 1;
        edge->y = get_edge_weight(this_r, this_g, this_b, other_r, other_g, other_b);
    }
}

// Kernel to decode graph
__global__
void decode(uint4 vertices[], char *image, uint num_vertices) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;
    for (uint pos = tid; pos < num_vertices; pos += num_threads) {
        uint img_pos = pos * CHANNEL_SIZE;

        char colour_1 = vertices[pos].y % 255;
        char colour_2 = (13*colour_1 + 101) % 255;
        char colour_3 = (13*colour_2 + 101) % 255;

        image[img_pos] = colour_1;
        image[img_pos + 1] = colour_2;
        image[img_pos + 2] = colour_3;
    }
}

#endif //FELZENSWALB_GRAPH_H
