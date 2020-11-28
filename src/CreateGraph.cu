//
// Created by akanksha on 20.11.20.
//
#include "CreateGraph.h"

__global__ void ImagetoGraph(cv::cuda::GpuMat Image, int *VertexList, edge *EdgeList, int *BitEdgeList, int pitch, int Channels){
    //https://stackoverflow.com/questions/23372262/access-pixels-in-gpumat/51189827
    //https://stackoverflow.com/questions/24613637/custom-kernel-gpumat-with-float
    int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;

    int rows = Image.rows;
    int cols = Image.cols;

    //TODO: Check if this needs to be fixed. Right now I am removing all border pixels
    if(i>rows-1)
        return;
    if(j>cols-1)
        return;

    //Add 8 neighbors of each pixel to the list of edges
    int PixIdx = i*cols + j;

    //TODO: Check if we really need 8 neighbors?
    //TODO: Check if this maps to some kind of deconvolution
    //TODO: Replace this with cublas??
    uint3 SrcPix, DestPix;

    VertexList[PixIdx] = 8*PixIdx; //VertexList stores the start of each index
    SrcPix.x = Image.data[ (i*Image.step) + j*Channels + 0];
    SrcPix.y = Image.data[ (i*Image.step) + j*Channels + 1];
    SrcPix.z = Image.data[ (i*Image.step) + j*Channels + 2];

    EdgeList[8*PixIdx].Vertex = i*cols + j-1; //Left
    DestPix.x = Image.data[ (i*Image.step) + (j-1)*Channels + 0];
    DestPix.y = Image.data[ (i*Image.step) + (j-1)*Channels + 1];
    DestPix.z = Image.data[ (i*Image.step) + (j-1)*Channels + 2];
    EdgeList[8*PixIdx].Weight = norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z);
    BitEdgeList[8*PixIdx] =(int(norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z))<<21)|(i*cols + j-1);
    //TODO: Find a way to retain double precision
    //TODO: Remove the EdgeList array

    EdgeList[8*PixIdx+1].Vertex = (i-1)*cols + j-1; //LeftTop
    DestPix.x = Image.data[ ((i-1)*Image.step) + (j-1)*Channels + 0];
    DestPix.y = Image.data[ ((i-1)*Image.step) + (j-1)*Channels + 1];
    DestPix.z = Image.data[ ((i-1)*Image.step) + (j-1)*Channels + 2];
    EdgeList[8*PixIdx+1].Weight = norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z);
    BitEdgeList[8*PixIdx+1] =(int(norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z))<<21)|((i-1)*cols + j-1);

    EdgeList[8*PixIdx+2].Vertex = (i-1)*cols + j; //Top
    DestPix.x = Image.data[ ((i-1)*Image.step) + j*Channels + 0];
    DestPix.y = Image.data[ ((i-1)*Image.step) + j*Channels + 1];
    DestPix.z = Image.data[ ((i-1)*Image.step) + j*Channels + 2];
    EdgeList[8*PixIdx+2].Weight = norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z);
    BitEdgeList[8*PixIdx+2] =(int(norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z))<<21)|((i-1)*cols + j);

    EdgeList[8*PixIdx+3].Vertex = (i-1)*cols + j+1; //TopRight
    DestPix.x = Image.data[ ((i-1)*Image.step) + (j+1)*Channels + 0];
    DestPix.y = Image.data[ ((i-1)*Image.step) + (j+1)*Channels + 1];
    DestPix.z = Image.data[ ((i-1)*Image.step) + (j+1)*Channels + 2];
    EdgeList[8*PixIdx+3].Weight = norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z);
    BitEdgeList[8*PixIdx+3] =(int(norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z))<<21)|((i-1)*cols + j+1);

    EdgeList[8*PixIdx+4].Vertex = i*cols + j+1; //Right
    DestPix.x = Image.data[ (i*Image.step) + (j+1)*Channels + 0];
    DestPix.y = Image.data[ (i*Image.step) + (j+1)*Channels + 1];
    DestPix.z = Image.data[ (i*Image.step) + (j+1)*Channels + 2];
    EdgeList[8*PixIdx+4].Weight = norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z);
    BitEdgeList[8*PixIdx+4] =(int(norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z))<<21)|(i*cols + j+1);

    EdgeList[8*PixIdx+5].Vertex = (i+1)*cols + j +1; //BottomRight
    DestPix.x = Image.data[ ((i+1)*Image.step) + (j+1)*Channels + 0];
    DestPix.y = Image.data[ ((i+1)*Image.step) + (j+1)*Channels + 1];
    DestPix.z = Image.data[ ((i+1)*Image.step) + (j+1)*Channels + 2];
    EdgeList[8*PixIdx+5].Weight = norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z);
    BitEdgeList[8*PixIdx+5] =(int(norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z))<<21)|((i+1)*cols + j+1);

    EdgeList[8*PixIdx+6].Vertex = (i+1)*cols + j; //Bottom
    DestPix.x = Image.data[ ((i-1)*Image.step) + j*Channels + 0];
    DestPix.y = Image.data[ ((i-1)*Image.step) + j*Channels + 1];
    DestPix.z = Image.data[ ((i-1)*Image.step) + j*Channels + 2];
    EdgeList[8*PixIdx+6].Weight = norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z);
    BitEdgeList[8*PixIdx+6] =(int(norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z))<<21)|((i+1)*cols + j);

    EdgeList[8*PixIdx+7].Vertex = (i+1)*cols + j-1; //BottomLeft
    DestPix.x = Image.data[ ((i+1)*Image.step) + (j-1)*Channels + 0];
    DestPix.y = Image.data[ ((i+1)*Image.step) + (j-1)*Channels + 1];
    DestPix.z = Image.data[ ((i+1)*Image.step) + (j-1)*Channels + 2];
    EdgeList[8*PixIdx+7].Weight = norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z);
    BitEdgeList[8*PixIdx+7] =(int(norm3df(SrcPix.x-DestPix.x,SrcPix.y-DestPix.y,SrcPix.z-DestPix.z))<<21)|((i+1)*cols + j-1);
}

