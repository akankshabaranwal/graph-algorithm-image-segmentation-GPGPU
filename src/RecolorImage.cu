//
// Created by akanksha on 19.12.20.
//

#include "RecolorImage.h"

using namespace cv::cuda;
using namespace cv;

__global__ void RandFloatToRandRGB(char* d_component_colours, float *d_component_colours_float, unsigned int n_numbers)
{
    unsigned int tid = blockIdx.x*1024 + threadIdx.x;
    if (tid < n_numbers) {
        float color = 255 *d_component_colours_float[tid];
        d_component_colours[tid] = (char) color;
    }
}
__global__ void InitPrevLevelComponents(unsigned int* d_prev_level_component, unsigned int no_of_rows, unsigned int no_of_cols)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < no_of_rows && col < no_of_cols) {
        d_prev_level_component[row * no_of_cols + col] = row * no_of_cols + col;
    }
}

__global__ void CreateLevelOutput(char *d_output_image, char *d_component_colours, uint32_t* d_level, unsigned int* d_prev_level_component, unsigned int no_of_rows, unsigned int no_of_cols)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < no_of_rows && col < no_of_cols) {

        unsigned int prev_component = d_prev_level_component[row * no_of_cols + col];
        unsigned int new_component = d_level[prev_component];

        int img_pos = 3 * (row * no_of_cols + col);
        int colour_pos = 3 * new_component;

        d_output_image[img_pos] = d_component_colours[colour_pos];
        d_output_image[img_pos + 1] = d_component_colours[colour_pos+1];
        d_output_image[img_pos + 2] = d_component_colours[colour_pos+2];

        d_prev_level_component[row * no_of_cols + col] = new_component;
    }
}

