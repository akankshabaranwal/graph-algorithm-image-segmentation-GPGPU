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
    int32_t Vertex;
    int32_t Weight;
};

int ImagetoGraphSerial(Mat image, edge *EdgeList, int32_t *VertexList, int32_t *BitEdgeList);
__global__ void ImagetoGraph(cv::cuda::GpuMat Image, int32_t *VertexList, edge *EdgeList, int32_t *BitEdgeList, int32_t *FlagList, int32_t pitch, int32_t channels);

#endif //FELZENSWALB_CREATEGRAPH_H