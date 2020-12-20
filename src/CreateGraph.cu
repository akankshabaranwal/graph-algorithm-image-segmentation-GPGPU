//
// Created by akanksha on 20.11.20.
//
#include "CreateGraph.h"

double dissimilarity(Mat image, uint32_t row1, uint32_t col1, uint32_t row2, uint32_t col2) {
    double dis = 0;
    Point3_<uchar>* u = image.ptr<Point3_<uchar> >(row1,col1);
    Point3_<uchar>* v = image.ptr<Point3_<uchar> >(row2,col2);
    dis = pow((u->x - v->x), 2) + pow((u->y - v->y), 2) + pow((u->z - v->z), 2);
    return sqrt(dis);
}

int ImagetoGraphSerial(Mat image, edge *EdgeList, uint32_t *VertexList, uint64_t *BitEdgeList)
{
    uint32_t cur_edge_idx, cur_vertex_idx, left_node, right_node, bottom_node, top_node;
    cur_edge_idx = 0;
    for(int i=0;i<image.rows;i++)
    {
        for(int j=0;j<image.cols;j++)
        {
            left_node = i * image.cols + j - 1;
            right_node = i * image.cols + j + 1;
            bottom_node = (i+1) * image.cols + j;
            top_node = (i - 1) * image.cols + j;
            //Add the index for VertexList
            cur_vertex_idx = i * image.cols + j;
            VertexList[cur_vertex_idx] = cur_edge_idx;
            if (j > 0){
                EdgeList[cur_edge_idx].Vertex = left_node;
                EdgeList[cur_edge_idx].Weight = dissimilarity(image, i, j, i, j - 1);
                //BitEdgeList[cur_edge_idx] = (EdgeList[cur_edge_idx].Weight * (2<<15)) + left_node;
                BitEdgeList[cur_edge_idx] = (EdgeList[cur_edge_idx].Weight <<32) | left_node;
                cur_edge_idx++;
            }
            if (j < image.cols - 1){
                EdgeList[cur_edge_idx].Vertex = right_node;
                EdgeList[cur_edge_idx].Weight = dissimilarity(image, i, j, i, j + 1);
                //BitEdgeList[cur_edge_idx] = (EdgeList[cur_edge_idx].Weight * (2<<15)) + right_node;
                BitEdgeList[cur_edge_idx] = (EdgeList[cur_edge_idx].Weight <<32) | right_node;
                cur_edge_idx++;
            }
            if (i < image.rows - 1){
                EdgeList[cur_edge_idx].Vertex = bottom_node;
                EdgeList[cur_edge_idx].Weight = dissimilarity(image, i, j, i+1, j);
                //BitEdgeList[cur_edge_idx] = (EdgeList[cur_edge_idx].Weight * (2<<15)) + bottom_node;
                BitEdgeList[cur_edge_idx] = (EdgeList[cur_edge_idx].Weight <<32) | bottom_node;
                cur_edge_idx++;
            }
            if (i > 0){
                EdgeList[cur_edge_idx].Vertex = top_node;
                EdgeList[cur_edge_idx].Weight = dissimilarity(image, i, j, i-1, j);
                //BitEdgeList[cur_edge_idx] = (EdgeList[cur_edge_idx].Weight * (2<<15)) + top_node;
                BitEdgeList[cur_edge_idx] = (EdgeList[cur_edge_idx].Weight <<32) | top_node;
                cur_edge_idx++;
            }
        }
    }
    return cur_edge_idx;
}

__global__ void ImagetoGraph(cv::cuda::GpuMat Image, int32_t *VertexList, edge *EdgeList, int32_t *BitEdgeList, int32_t *FlagList, int32_t pitch, int32_t Channels){

    int32_t i = blockIdx.x*blockDim.x + threadIdx.x +1;
    int32_t j = blockIdx.y*blockDim.y + threadIdx.y +1;

    int32_t rows = Image.rows;
    int32_t cols = Image.cols;

    //TODO: Check if this needs to be fixed. Right now I am removing all border pixels
    if(i>rows-1)
        return;
    if(j>cols-1)
        return;

    //Add 8 neighbors of each pixel to the list of edges
    int32_t PixIdx = i*cols + j;
    if(PixIdx >= 60000)
    {
        printf("ERROR: Something went wrong: %d i, %d j\n", i, j);
    }
    //TODO: Check if we really need 8 neighbors?
    int32_t SrcPixX, SrcPixY, SrcPixZ;
    int32_t DestPixX, DestPixY, DestPixZ;
    int32_t DiffX, DiffY, DiffZ;

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
    EdgeList[8*PixIdx].Weight = int32_t(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx] = (EdgeList[8*PixIdx].Weight * (2<<15)) + EdgeList[8*PixIdx].Vertex;

    EdgeList[8*PixIdx+1].Vertex = (i-1)*cols + j-1; //LeftTop
    DestPixX = Image.data[ ((i-1)*Image.step) + (j-1)*Channels + 0];
    DestPixY = Image.data[ ((i-1)*Image.step) + (j-1)*Channels + 1];
    DestPixZ = Image.data[ ((i-1)*Image.step) + (j-1)*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx+1].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx+1] = (EdgeList[8*PixIdx+1].Weight*(2<<15)) + EdgeList[8*PixIdx+1].Vertex;

    EdgeList[8*PixIdx+2].Vertex = (i-1)*cols + j; //Top
    DestPixX = Image.data[ ((i-1)*Image.step) + j*Channels + 0];
    DestPixY = Image.data[ ((i-1)*Image.step) + j*Channels + 1];
    DestPixZ = Image.data[ ((i-1)*Image.step) + j*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx+2].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx+2] = (EdgeList[8*PixIdx+2].Weight*(2<<15)) + EdgeList[8*PixIdx+2].Vertex;

    EdgeList[8*PixIdx+3].Vertex = (i-1)*cols + j+1; //TopRight
    DestPixX = Image.data[ ((i-1)*Image.step) + (j+1)*Channels + 0];
    DestPixY = Image.data[ ((i-1)*Image.step) + (j+1)*Channels + 1];
    DestPixZ = Image.data[ ((i-1)*Image.step) + (j+1)*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx+3].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx+3] = (EdgeList[8*PixIdx+3].Weight*(2<<15)) + EdgeList[8*PixIdx+3].Vertex;

    EdgeList[8*PixIdx+4].Vertex = i*cols + j+1; //Right
    DestPixX = Image.data[ (i*Image.step) + (j+1)*Channels + 0];
    DestPixY = Image.data[ (i*Image.step) + (j+1)*Channels + 1];
    DestPixZ = Image.data[ (i*Image.step) + (j+1)*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx+4].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx+4] = (EdgeList[8*PixIdx+4].Weight*(2<<15)) + EdgeList[8*PixIdx+4].Vertex;

    EdgeList[8*PixIdx+5].Vertex = (i+1)*cols + j +1; //BottomRight
    DestPixX = Image.data[ ((i+1)*Image.step) + (j+1)*Channels + 0];
    DestPixY = Image.data[ ((i+1)*Image.step) + (j+1)*Channels + 1];
    DestPixZ = Image.data[ ((i+1)*Image.step) + (j+1)*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx+5].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx+5] = (EdgeList[8*PixIdx+5].Weight*(2<<15)) + EdgeList[8*PixIdx+5].Vertex;

    EdgeList[8*PixIdx+6].Vertex = (i+1)*cols + j; //Bottom
    DestPixX = Image.data[ ((i-1)*Image.step) + j*Channels + 0];
    DestPixY = Image.data[ ((i-1)*Image.step) + j*Channels + 1];
    DestPixZ = Image.data[ ((i-1)*Image.step) + j*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx+6].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx+6] =(EdgeList[8*PixIdx+6].Weight*(2<<15))+ EdgeList[8*PixIdx+6].Vertex;

    EdgeList[8*PixIdx+7].Vertex = (i+1)*cols + j-1; //BottomLeft
    DestPixX = Image.data[ ((i+1)*Image.step) + (j-1)*Channels + 0];
    DestPixY = Image.data[ ((i+1)*Image.step) + (j-1)*Channels + 1];
    DestPixZ = Image.data[ ((i+1)*Image.step) + (j-1)*Channels + 2];
    DiffX = DestPixX - SrcPixX;
    DiffY = DestPixY - SrcPixY;
    DiffZ = DestPixZ - SrcPixZ;
    EdgeList[8*PixIdx+7].Weight = int(sqrtf(DiffX*DiffX + DiffY*DiffY + DiffZ*DiffZ));
    BitEdgeList[8*PixIdx+7] =(EdgeList[8*PixIdx+7].Weight *(2<<15)) + EdgeList[8*PixIdx+7].Vertex;
}