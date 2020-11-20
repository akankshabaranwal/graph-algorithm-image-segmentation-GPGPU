//
// Created by akanksha on 20.11.20.
//

#ifndef FELZENSZWALB_CREATEGRAPH_H
#define FELZENSZWALB_CREATEGRAPH_H

#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>

class edge{
    int Vertex;
    float Weight;
}E;
__global__ void ImagetoGraph(cv::cuda::GpuMat Image, int *VertexList, uint64_t *EdgeList);
// Check if the graph formation can be implemented using deconvolution??


#endif //FELZENSWALB_CREATEGRAPH_H
