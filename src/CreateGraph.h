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


//TODO: Fix the weight from int to float
struct edge{
  public:
    uint32_t Vertex;
    uint64_t Weight;
};

int ImagetoGraphSerial(Mat image, edge *EdgeList, uint32_t *VertexList, uint64_t *BitEdgeList);
void ImagetoGraphParallelStream(Mat &image, uint32_t *d_vertex,uint32_t *d_edge, uint64_t *d_weight);
__global__ void ImagetoGraph(cv::cuda::GpuMat Image, int32_t *VertexList, edge *EdgeList, int32_t *BitEdgeList, int32_t *FlagList, int32_t pitch, int32_t channels);
void SetImageGridThreadLen(int no_of_rows, int no_of_cols, int no_of_vertices, dim3* encode_threads, dim3* encode_blocks);
void SetGridThreadLen(int number, int *num_of_blocks, int *num_of_threads_per_block);

#endif //FELZENSWALB_CREATEGRAPH_H