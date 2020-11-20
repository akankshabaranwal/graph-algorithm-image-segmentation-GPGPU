//
// Created by akanksha on 20.11.20.
//

#ifndef FELZENSZWALB_CREATEGRAPH_H
#define FELZENSZWALB_CREATEGRAPH_H

#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>

struct edge{
  public:
    int Vertex;
    float Weight;
};
__global__ void ImagetoGraph(cv::cuda::GpuMat Image, int *VertexList, edge *EdgeList);
// Check if the graph formation can be implemented using deconvolution??


#endif //FELZENSWALB_CREATEGRAPH_H
