//
// Created by akanksha on 20.11.20.
//

#ifndef FELZENSZWALB_CREATEGRAPH_H
#define FELZENSZWALB_CREATEGRAPH_H

#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <iostream>

using namespace cv::cuda;
using namespace cv;

__global__ void createCornerGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, uint64_t *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch);
__global__ void createFirstRowGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, uint64_t *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch);
__global__ void createLastRowGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, uint64_t *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch);
__global__ void createFirstColumnGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, uint64_t *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch);
__global__ void createLastColumnGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, uint64_t *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch);
__global__ void createInnerGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, uint64_t *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch);

void ImagetoGraphParallelStream(Mat &image, uint32_t *d_vertex,uint32_t *d_edge, uint64_t *d_weight);
void SetImageGridThreadLen(int no_of_rows, int no_of_cols, int no_of_vertices, dim3* encode_threads, dim3* encode_blocks);
void SetGridThreadLen(int number, int *num_of_blocks, int *num_of_threads_per_block);

#endif //FELZENSWALB_CREATEGRAPH_H