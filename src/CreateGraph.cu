//
// Created by akanksha on 20.11.20.
//
#include "CreateGraph.h"
__global__ void ImagetoGraph(cv::cuda::GpuMat Image, int *VertexList, edge *EdgeList){

    int i = blockIdx.x*blockDim.x + threadIdx.x +1;
    int j = blockIdx.y*blockDim.y + threadIdx.y+1;
    //TODO: Check if this needs to be fixed. Right now I am removing all border pixels
    if(i>Image.rows-1)
        return;
    if(j>Image.cols-1)
        return;

    //Add 8 neighbors of each pixel to the list of edges
    int PixIdx = i*Image.cols + j;

    //TODO: Check if we really need 8 neighbors?
    //TODO: Check if this maps to some kind of convolution

    VertexList[PixIdx] = 8*PixIdx; //VertexList stores the start of each index
    EdgeList[8*PixIdx].Vertex = i*Image.cols + j-1; //Left
    EdgeList[8*PixIdx+1].Vertex = (i-1)*Image.cols + j-1; //LeftTop
    EdgeList[8*PixIdx+2].Vertex = (i-1)*Image.cols + j; //Top
    EdgeList[8*PixIdx+3].Vertex = (i-1)*Image.cols + j+1; //TopRight
    EdgeList[8*PixIdx+4].Vertex = i*Image.cols + j+1; //Right
    EdgeList[8*PixIdx+5].Vertex = (i+1)*Image.cols + j +1; //BottomRight
    EdgeList[8*PixIdx+6].Vertex = (i+1)*Image.cols + j; //Bottom
    EdgeList[8*PixIdx+7].Vertex = (i+1)*Image.cols + j-1; //BottomLeft
}

