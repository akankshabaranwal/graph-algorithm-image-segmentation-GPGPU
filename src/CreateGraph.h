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
    int Vertex;
    int Weight;
};
__global__ void ImagetoGraph(cv::cuda::GpuMat Image, int *VertexList, edge *EdgeList, int *BitEdgeList, int pitch, int channels);
// Check if the graph formation can be implemented using deconvolution??

#endif //FELZENSWALB_CREATEGRAPH_H