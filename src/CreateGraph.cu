//
// Created by akanksha on 20.11.20.
//
#include "CreateGraph.h"
__global__ void ImagetoGraph(cv::cuda::GpuMat Image, int *VertexList, uint64_t *EdgeList){

    int i = blockIdx.x*blockDim.x + threadIdx.x +1;
    int j = blockIdx.y*blockDim.y + threadIdx.y+1;
    //TODO: Check if this needs to be fixed. Right now I am removing all border pixels
    if(i>Image.rows-1)
        return;
    if(j>Image.cols-1)
        return;

    //Add 8 neighbors of each pixel to the list of edges
    int PixIdx = i*Image.cols + j;
    int Left = i*Image.cols + j-1;
    int LeftTop = (i-1)*Image.cols + j-1;
    int Top = (i-1)*Image.cols + j;
    int TopRight = (i-1)*Image.cols + j+1;
    int Right = i*Image.cols + j+1;
    int BottomRight = (i+1)*Image.cols + j +1;
    int Bottom=(i+1)*Image.cols + j;
    int BottomLeft=(i+1)*Image.cols + j-1;

    //TODO: Replace the above variables with directly in the values being computed. No need for these variables.
    //TODO: Check if we really need 8 neighbors?
    //TODO: Check if this maps to some kind of convolution

    VertexList[PixIdx] = 8*PixIdx; //VertexList stores the start of each index
    EdgeList[8*PixIdx] = Left;
    EdgeList[8*PixIdx+1] = LeftTop;
    EdgeList[8*PixIdx+2] = Top;
    EdgeList[8*PixIdx+3] = TopRight;
    EdgeList[8*PixIdx+4] = Right;
    EdgeList[8*PixIdx+5] = BottomRight;
    EdgeList[8*PixIdx+6] = Bottom;
    EdgeList[8*PixIdx+7] = BottomLeft;
}

