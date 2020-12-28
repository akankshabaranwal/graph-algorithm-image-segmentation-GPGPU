//
// Created by akanksha on 19.12.20.
//

#ifndef FELZENSZWALB_RECOLORIMAGE_H
#define FELZENSZWALB_RECOLORIMAGE_H
#include <iostream>
#include <vector>
#include "CreateGraph.h"
// Curand stuff
#include <cuda.h>
#include <curand.h>
__global__ void RandFloatToRandRGB(char* d_component_colours, float *d_component_colours_float, unsigned int n_numbers);
__global__ void InitPrevLevelComponents(unsigned int* d_prev_level_component, unsigned int no_of_rows, unsigned int no_of_cols);
__global__ void CreateLevelOutput(char *d_output_image, char *d_component_colours, uint32_t* d_level, unsigned int* d_prev_level_component, unsigned int no_of_rows, unsigned int no_of_cols);


#endif //FELZENSZWALB_RECOLORIMAGE_H
