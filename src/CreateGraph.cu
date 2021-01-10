//
// Created by akanksha on 20.11.20.
//
#include "CreateGraph.h"

////////////////////////////////////////////////////////////////////////////////////////////
// Graph creation kernels. By Amory
////////////////////////////////////////////////////////////////////////////////////////////
__global__ void createCornerGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, uint64_t *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch)
{
    unsigned int tid = blockIdx.x*1024 + threadIdx.x;
    if (tid < 4) {
        unsigned int row = 0;
        unsigned int col = 0;
        unsigned int write_offset = 0;

        if (tid == 1) {
            col = no_of_cols - 1;
            write_offset = 3 * (no_of_cols - 1) - 1;
        }
        if (tid == 2) {
            row = no_of_rows - 1;
            write_offset = 4 + 6 * (no_of_rows-2) + 3 * (no_of_cols-2) + 4 * (no_of_rows-2) * (no_of_cols-2);
        }
        if (tid == 3) {
            col = no_of_cols - 1;
            row = no_of_rows - 1;
            write_offset = 6 + 6 * (no_of_rows-2) + 6 * (no_of_cols-2) + 4 * (no_of_rows-2) * (no_of_cols-2);
        }

        unsigned int left_node = row * no_of_cols + col - 1;
        unsigned int right_node = row * no_of_cols + col + 1;
        unsigned int top_node = (row - 1) * no_of_cols + col;
        unsigned int bottom_node = (row+1) * no_of_cols + col;

        unsigned int this_img_idx = row * pitch + col * 3;
        unsigned char this_r = image[this_img_idx];
        unsigned char this_g = image[this_img_idx + 1];
        unsigned char this_b = image[this_img_idx + 2];

        unsigned char other_r;
        unsigned char other_g;
        unsigned char other_b;
        unsigned int other_img_idx;
        double distance;

        unsigned long cur_vertex_idx = row * no_of_cols + col;
        d_vertex[cur_vertex_idx] = write_offset;

        // Left node
        if (tid == 1 || tid == 3) {
            d_edge[write_offset] = left_node;

            other_img_idx = row * pitch + (col - 1) * 3;
            other_r = image[other_img_idx];
            other_g = image[other_img_idx + 1];
            other_b = image[other_img_idx + 2];
            distance = 8.0*sqrtf(powf((this_r - other_r), 2) + powf((this_g - other_g), 2) + powf((this_b - other_b), 2.0));
            d_weight[write_offset] = (unsigned int) round(distance);
        }

        // Right node
        if (tid == 0 || tid == 2) {
            d_edge[write_offset] = right_node;

            other_img_idx = row * pitch + (col + 1) * 3;
            other_r = image[other_img_idx];
            other_g = image[other_img_idx + 1];
            other_b = image[other_img_idx + 2];
            distance = 8.0*sqrtf(powf((this_r - other_r), 2) + powf((this_g - other_g), 2) + powf((this_b - other_b), 2.0));
            d_weight[write_offset] = (unsigned int) round(distance);
        }

        // Bottom node
        if (tid == 0 || tid == 1) {
            d_edge[write_offset+1] = bottom_node;

            other_img_idx = (row+1) * pitch + col * 3;
            other_r = image[other_img_idx];
            other_g = image[other_img_idx + 1];
            other_b = image[other_img_idx + 2];
            distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
            d_weight[write_offset+1] = (unsigned int) round(distance);
        }

        // Top node
        if (tid == 2 || tid == 3) {
            d_edge[write_offset+1] = top_node;

            other_img_idx = (row-1) * pitch + col * 3;
            other_r = image[other_img_idx];
            other_g = image[other_img_idx + 1];
            other_b = image[other_img_idx + 2];
            distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
            d_weight[write_offset+1] = (unsigned int) round(distance);
        }
    }
}


__global__ void createFirstRowGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, uint64_t *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch)
{
    unsigned int row = 0;
    unsigned int col = blockIdx.x*1024 + threadIdx.x;

    if (col > 0 && col < no_of_cols - 1) {
        unsigned int left_node = row * no_of_cols + col - 1;
        unsigned int right_node = row * no_of_cols + col + 1;
        unsigned int bottom_node = (row+1) * no_of_cols + col;

        unsigned int this_img_idx = row * pitch + col * 3;
        unsigned char this_r = image[this_img_idx];
        unsigned char this_g = image[this_img_idx + 1];
        unsigned char this_b = image[this_img_idx + 2];

        unsigned int write_offset = 2 + (col-1) * 3;

        unsigned char other_r;
        unsigned char other_g;
        unsigned char other_b;
        unsigned int other_img_idx;
        double distance;

        unsigned long cur_vertex_idx = row * no_of_cols + col;
        d_vertex[cur_vertex_idx] = write_offset;

        // Left node
        d_edge[write_offset] = left_node;

        other_img_idx = row * pitch + (col - 1) * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset] = (unsigned int) round(distance);

        // Right node
        d_edge[write_offset+1] = right_node;

        other_img_idx = row * pitch + (col + 1) * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset+1] = (unsigned int) round(distance);

        // Bottom node
        d_edge[write_offset+2] = bottom_node;

        other_img_idx = (row+1) * pitch + col * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset+2] = (unsigned int) round(distance);
    }
}

__global__ void createLastRowGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, uint64_t *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch)
{
    unsigned int row = no_of_rows-1;;
    unsigned int col = blockIdx.x*1024 + threadIdx.x;

    if (col > 0 && col < no_of_cols - 1) {
        unsigned int left_node = row * no_of_cols + col - 1;
        unsigned int right_node = row * no_of_cols + col + 1;
        unsigned int top_node = (row - 1) * no_of_cols + col;

        unsigned int this_img_idx = row * pitch + col * 3;
        unsigned char this_r = image[this_img_idx];
        unsigned char this_g = image[this_img_idx + 1];
        unsigned char this_b = image[this_img_idx + 2];

        unsigned int first_row_offset = 4 + 3 * (no_of_cols-2);
        unsigned int extra_cur_row_offset = 3 + (row-1) * (6 + 4 * (no_of_cols-2));
        unsigned int extra_cur_col_offset = 3 * (col-1);
        unsigned int write_offset = first_row_offset + extra_cur_row_offset + extra_cur_col_offset - 1;

        unsigned char other_r;
        unsigned char other_g;
        unsigned char other_b;
        unsigned int other_img_idx;
        double distance;

        unsigned long cur_vertex_idx = row * no_of_cols + col;
        d_vertex[cur_vertex_idx] = write_offset;

        // Left node
        d_edge[write_offset] = left_node;

        other_img_idx = row * pitch + (col - 1) * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset] = (unsigned int) round(distance);

        // Right node
        d_edge[write_offset+1] = right_node;

        other_img_idx = row * pitch + (col + 1) * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset+1] = (unsigned int) round(distance);

        // Top node
        d_edge[write_offset+2] = top_node;

        other_img_idx = (row-1) * pitch + col * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset+2] = (unsigned int) round(distance);
    }
}

__global__ void createFirstColumnGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, uint64_t *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = 0;

    if (row > 0 && row < no_of_rows - 1) {
        unsigned int right_node = row * no_of_cols + col + 1;
        unsigned int top_node = (row - 1) * no_of_cols + col;
        unsigned int bottom_node = (row+1) * no_of_cols + col;

        unsigned int this_img_idx = row * pitch + col * 3;
        unsigned char this_r = image[this_img_idx];
        unsigned char this_g = image[this_img_idx + 1];
        unsigned char this_b = image[this_img_idx + 2];

        unsigned int first_row_offset = 4 + 3 * (no_of_cols-2);
        unsigned int extra_cur_row_offset = (row-1) * (6 + 4 * (no_of_cols-2));
        unsigned int write_offset = first_row_offset + extra_cur_row_offset;

        unsigned char other_r;
        unsigned char other_g;
        unsigned char other_b;
        unsigned int other_img_idx;
        double distance;

        unsigned long cur_vertex_idx = row * no_of_cols + col;
        d_vertex[cur_vertex_idx] = write_offset;

        // Right node
        d_edge[write_offset] = right_node;

        other_img_idx = row * pitch + (col + 1) * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset] = (unsigned int) round(distance);

        // Bottom node
        d_edge[write_offset+1] = bottom_node;

        other_img_idx = (row+1) * pitch + col * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset+1] = (unsigned int) round(distance);

        // Top node
        d_edge[write_offset+2] = top_node;

        other_img_idx = (row-1) * pitch + col * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset+2] = (unsigned int) round(distance);
    }
}

__global__ void createLastColumnGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, uint64_t *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = no_of_cols - 1;

    if (row > 0 && row < no_of_rows - 1) {
        unsigned int left_node = row * no_of_cols + col - 1;
        unsigned int top_node = (row - 1) * no_of_cols + col;
        unsigned int bottom_node = (row+1) * no_of_cols + col;

        unsigned int this_img_idx = row * pitch + col * 3;
        unsigned char this_r = image[this_img_idx];
        unsigned char this_g = image[this_img_idx + 1];
        unsigned char this_b = image[this_img_idx + 2];

        unsigned int first_row_offset = 4 + 3 * (no_of_cols-2);
        unsigned int extra_cur_row_offset = 3 + (row-1) * (6 + 4 * (no_of_cols-2));
        unsigned int extra_cur_col_offset = 4 * (col-1);
        unsigned int write_offset = first_row_offset + extra_cur_row_offset + extra_cur_col_offset;

        unsigned char other_r;
        unsigned char other_g;
        unsigned char other_b;
        unsigned int other_img_idx;
        double distance;

        unsigned long cur_vertex_idx = row * no_of_cols + col;
        d_vertex[cur_vertex_idx] = write_offset;

        // Left node
        d_edge[write_offset] = left_node;

        other_img_idx = row * pitch + (col - 1) * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset] = (unsigned int) round(distance);

        // Bottom node
        d_edge[write_offset+1] = bottom_node;

        other_img_idx = (row+1) * pitch + col * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset+1] = (unsigned int) round(distance);

        // Top node
        d_edge[write_offset+2] = top_node;

        other_img_idx = (row-1) * pitch + col * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset+2] = (unsigned int) round(distance);
    }
}

__global__ void createInnerGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, uint64_t *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row > 0 && col > 0 && row < no_of_rows - 1 && col < no_of_cols - 1) {
        unsigned int left_node = row * no_of_cols + col - 1;
        unsigned int right_node = row * no_of_cols + col + 1;
        unsigned int top_node = (row - 1) * no_of_cols + col;
        unsigned int bottom_node = (row+1) * no_of_cols + col;

        unsigned int this_img_idx = row * pitch + col * 3;
        unsigned char this_r = image[this_img_idx];
        unsigned char this_g = image[this_img_idx + 1];
        unsigned char this_b = image[this_img_idx + 2];

        unsigned int first_row_offset = 4 + 3 * (no_of_cols-2);
        unsigned int extra_cur_row_offset = 3 + (row-1) * (6 + 4 * (no_of_cols-2));
        unsigned int extra_cur_col_offset = 4 * (col-1);
        unsigned int write_offset = first_row_offset + extra_cur_row_offset + extra_cur_col_offset;

        unsigned char other_r;
        unsigned char other_g;
        unsigned char other_b;
        unsigned int other_img_idx;
        double distance;

        unsigned long cur_vertex_idx = row * no_of_cols + col;
        d_vertex[cur_vertex_idx] = write_offset;

        // Left node
        d_edge[write_offset] = left_node;

        other_img_idx = row * pitch + (col - 1) * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset] = (unsigned int) round(distance);

        // Right node
        d_edge[write_offset+1] = right_node;

        other_img_idx = row * pitch + (col + 1) * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset+1] = (unsigned int) round(distance);

        // Bottom node
        d_edge[write_offset+2] = bottom_node;

        other_img_idx = (row+1) * pitch + col * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset+2] = (unsigned int) round(distance);

        // Top node
        d_edge[write_offset+3] = top_node;

        other_img_idx = (row-1) * pitch + col * 3;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
        distance = 8.0*sqrtf(powf((this_r - other_r), 2.0) + powf((this_g - other_g), 2.0) + powf((this_b - other_b), 2.0));
        d_weight[write_offset+3] = (unsigned int) round(distance);
    }
}

////////////////////////////////////////////////
// Helper functions to set the grid sizes
////////////////////////////////////////////////
void SetGridThreadLen(int number, int *num_of_blocks, int *num_of_threads_per_block)
{
    *num_of_blocks = 1;
    *num_of_threads_per_block = number;

    //Make execution Parameters according to the number of nodes
    //Distribute threads across multiple Blocks if necessary
    if(number>1024)
    {
        *num_of_blocks = (int)ceil(number/(double)1024);
        *num_of_threads_per_block = 1024;
    }
}

void SetImageGridThreadLen(int no_of_rows, int no_of_cols, int no_of_vertices, dim3* encode_threads, dim3* encode_blocks)
{
    if (no_of_vertices < 1024) {
        encode_threads->x = no_of_rows;
        encode_threads->y = no_of_cols;
        encode_blocks->x = 1;
        encode_blocks->y = 1;
    } else {
        encode_threads->x = 32;
        encode_threads->y = 32;
        encode_blocks->x = no_of_rows / 32 + 1;
        encode_blocks->y = no_of_cols / 32 + 1;
    }
}