//
// Created by akanksha on 20.11.20.
//
#include "CreateGraph.h"

__global__ void ImagetoGraph(cv::cuda::GpuMat Image, int *VertexList, edge *EdgeList, int *BitEdgeList, int pitch, int Channels){
    //https://stackoverflow.com/questions/23372262/access-pixels-in-gpumat/51189827
    //https://stackoverflow.com/questions/24613637/custom-kernel-gpumat-with-float
    int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    //https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html: For bitwise shift etc
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
    //uint3 SrcPix, DestPix;
    float SrcPixX, SrcPixY, SrcPixZ;
    float DestPixX, DestPixY, DestPixZ;
    float DiffX, DiffY, DiffZ;

    //Using 16 bits for Weight and 16 for vertex id

    VertexList[PixIdx] = 8*PixIdx; //VertexList stores the start of each index
    SrcPixX = Image.data[ (i*Image.step) + j*Channels + 0];
    SrcPixY = Image.data[ (i*Image.step) + j*Channels + 1];
    SrcPixZ = Image.data[ (i*Image.step) + j*Channels + 2];

    //TODO: Remove the weight parameter from edgelist array
    EdgeList[8*PixIdx].Vertex = i*cols + j-1; //Left
    DestPixX = Image.data[ (i*Image.step) + (j-1)*Channels + 0];
    DestPixY = Image.data[ (i*Image.step) + (j-1)*Channels + 1];
    DestPixZ = Image.data[ (i*Image.step) + (j-1)*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx] = (EdgeList[8*PixIdx].Weight * (2<<16)) + EdgeList[8*PixIdx].Vertex;

    EdgeList[8*PixIdx+1].Vertex = (i-1)*cols + j-1; //LeftTop
    DestPixX = Image.data[ ((i-1)*Image.step) + (j-1)*Channels + 0];
    DestPixY = Image.data[ ((i-1)*Image.step) + (j-1)*Channels + 1];
    DestPixZ = Image.data[ ((i-1)*Image.step) + (j-1)*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx+1].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx+1] = (EdgeList[8*PixIdx+1].Weight*(2<<16)) + EdgeList[8*PixIdx+1].Vertex;

    EdgeList[8*PixIdx+2].Vertex = (i-1)*cols + j; //Top
    DestPixX = Image.data[ ((i-1)*Image.step) + j*Channels + 0];
    DestPixY = Image.data[ ((i-1)*Image.step) + j*Channels + 1];
    DestPixZ = Image.data[ ((i-1)*Image.step) + j*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx+2].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx+2] = (EdgeList[8*PixIdx+2].Weight*(2<<16)) + EdgeList[8*PixIdx+2].Vertex;

    EdgeList[8*PixIdx+3].Vertex = (i-1)*cols + j+1; //TopRight
    DestPixX = Image.data[ ((i-1)*Image.step) + (j+1)*Channels + 0];
    DestPixY = Image.data[ ((i-1)*Image.step) + (j+1)*Channels + 1];
    DestPixZ = Image.data[ ((i-1)*Image.step) + (j+1)*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx+3].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx+3] = (EdgeList[8*PixIdx+3].Weight*(2<<16)) + EdgeList[8*PixIdx+3].Vertex;

    EdgeList[8*PixIdx+4].Vertex = i*cols + j+1; //Right
    DestPixX = Image.data[ (i*Image.step) + (j+1)*Channels + 0];
    DestPixY = Image.data[ (i*Image.step) + (j+1)*Channels + 1];
    DestPixZ = Image.data[ (i*Image.step) + (j+1)*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx+4].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx+4] = (EdgeList[8*PixIdx+4].Weight*(2<<16)) + EdgeList[8*PixIdx+4].Vertex;

    EdgeList[8*PixIdx+5].Vertex = (i+1)*cols + j +1; //BottomRight
    DestPixX = Image.data[ ((i+1)*Image.step) + (j+1)*Channels + 0];
    DestPixY = Image.data[ ((i+1)*Image.step) + (j+1)*Channels + 1];
    DestPixZ = Image.data[ ((i+1)*Image.step) + (j+1)*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx+5].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx+5] = (EdgeList[8*PixIdx+5].Weight*(2<<16)) + EdgeList[8*PixIdx+5].Vertex;

    EdgeList[8*PixIdx+6].Vertex = (i+1)*cols + j; //Bottom
    DestPixX = Image.data[ ((i-1)*Image.step) + j*Channels + 0];
    DestPixY = Image.data[ ((i-1)*Image.step) + j*Channels + 1];
    DestPixZ = Image.data[ ((i-1)*Image.step) + j*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx+6].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx+6] =(EdgeList[8*PixIdx+6].Weight*(2<<16))+ EdgeList[8*PixIdx+6].Vertex;

    EdgeList[8*PixIdx+7].Vertex = (i+1)*cols + j-1; //BottomLeft
    DestPixX = Image.data[ ((i+1)*Image.step) + (j-1)*Channels + 0];
    DestPixY = Image.data[ ((i+1)*Image.step) + (j-1)*Channels + 1];
    DestPixZ = Image.data[ ((i+1)*Image.step) + (j-1)*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx+7].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx+7] =(EdgeList[8*PixIdx+7].Weight *(2<<16)) + EdgeList[8*PixIdx+7].Vertex;
}