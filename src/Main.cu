#include <iostream>
#include <cuda_runtime_api.h>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>
#include "CreateGraph.h"
#include "moderngpu.cuh"		// Include all MGPU kernels.
#include "FastMST.h"

using namespace cv;
using namespace cv::cuda;
using namespace mgpu;

int main(int argc, char **argv)
{
    Mat image, output;
    GpuMat dev_image, dev_output;

    image = imread("data/beach.png", IMREAD_COLOR);
    printf("Size of image obtained is: Rows: %d, Columns: %d, Pixels: %d\n", image.rows, image.cols, image.rows * image.cols);
    //TODO: Add checker for image size depending on the bits decided for representing edge weight and vertex index
    dev_image.upload(image);

    Ptr<Filter> filter = createGaussianFilter(CV_8UC3, CV_8UC3, Size(5, 5), 1.0);
    filter->apply(dev_image, dev_output);

    //Graph parameters
    int numVertices = image.rows*image.cols;
    int numEdges= (image.rows)*(image.cols)*8;

    //Convert image to graph
    int32_t *VertexList, *BitEdgeList, *FlagList, *OutList, *NWE, *Successor, *Representative, *Vertex;
    edge *EdgeList;

    cudaMallocManaged(&VertexList,numVertices*sizeof(int32_t));
    cudaMallocManaged(&FlagList,numVertices*sizeof(int32_t));
    cudaMallocManaged(&OutList,numVertices*sizeof(int32_t));
    cudaMallocManaged(&EdgeList,numEdges*sizeof(edge));
    cudaMallocManaged(&BitEdgeList,numEdges*sizeof(int32_t));
    cudaMallocManaged(&NWE,numVertices*sizeof(int32_t));
    cudaMallocManaged(&Successor,numVertices*sizeof(int32_t));

    cudaMallocManaged(&Representative,numVertices*sizeof(int32_t));
    cudaMallocManaged(&Vertex,numVertices*sizeof(int32_t));
    int *Flag2;
    cudaMallocManaged(&Flag2,numVertices*sizeof(int32_t));
    int *SuperVertexId;
    cudaMallocManaged(&SuperVertexId,numVertices*sizeof(int32_t));
    int *flag;
    cudaMallocManaged(&flag,numVertices*sizeof(int32_t));
    int *uid;
    cudaMallocManaged(&uid,numVertices*sizeof(int32_t));

    dim3 threadsPerBlock(32,32);
    int BlockX = image.rows/threadsPerBlock.x;
    int BlockY = image.cols/threadsPerBlock.y;
    dim3 numBlocks(BlockX, BlockY);
    cudaDeviceSynchronize();
    for(int i =0;i<numEdges;i++)
    {
    EdgeList[i].Weight=0;
    }
    dev_output.download(output);
    //ImagetoGraph<<<numBlocks,threadsPerBlock>>>(dev_image, VertexList, EdgeList, BitEdgeList, FlagList, dev_image.step, 3);
    numEdges = ImagetoGraphSerial(image, EdgeList, VertexList, BitEdgeList);
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        printf("CUDA Error in ImagetoGraph function call: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    ContextPtr context = CreateCudaDevice(argc, argv, true);
    //For the first iteration VertexList and FlagList are exactly same
    //Maybe we don't need separate OutList and NWE arrays
    SegmentedReduction(*context, VertexList, BitEdgeList, OutList, NWE, numEdges, numVertices);
    int numthreads = 1024;
    int numBlock = numVertices/numthreads;
    FindSuccessorArray<<<numBlock,numthreads>>>(Successor, NWE, numVertices);
    err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        printf("CUDA Error in FindSuccessorArray function call: %s\n", cudaGetErrorString(err));
    }
    RemoveCycles<<<numBlock,numthreads>>>(Successor, numVertices);
    err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        printf("CUDA Error in RemoveCycles function call: %s\n", cudaGetErrorString(err));
    }
    PropagateRepresentativeVertices(Successor, numVertices);
    SortedSplit(Representative, Vertex, Successor, Flag2, numVertices);
    RemoveSelfEdges<<<numBlock,numthreads>>>(SuperVertexId, Vertex, Flag2, numVertices);
    //TODO: This needs to be moved to before
    MarkSegments<<<numBlock,numthreads>>>(flag, VertexList,numVertices);

   //10.2 Not sure why we have this?
    CreateUid(uid, flag, numVertices); //Maybe this Uid is not required. VerticesList has the same redundant info?
    RemoveEdge<<<numBlock,numthreads>>>(BitEdgeList, numEdges, uid, SuperVertexId);

    return 0;
}
