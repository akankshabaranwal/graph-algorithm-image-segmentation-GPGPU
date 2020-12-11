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
    int numEdges= (image.rows)*(image.cols)*4;

    //Convert image to graph
    int32_t *VertexList, *BitEdgeList, *FlagList, *OutList, *NWE, *Successor, *Representative, *Vertex;
    edge *EdgeList;

    //Allocating memory
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
    cudaDeviceSynchronize();//FIXME: Need to check where all this synchronize call is needed
    ContextPtr context = CreateCudaDevice(argc, argv, true);
    cudaError_t err = cudaGetLastError();

    //FIXME: Make this initialization run in parallel?
    //TODO: Figure out if this initialization is required??
    for(int i =0;i<numEdges;i++)
    {
    EdgeList[i].Weight=0;
    }

    dev_output.download(output);
    /*ImagetoGraph<<<numBlocks,threadsPerBlock>>>(dev_image, VertexList, EdgeList, BitEdgeList, FlagList, dev_image.step, 3);

    if ( err != cudaSuccess )
    {
        printf("CUDA Error in ImagetoGraph function call: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();*/

    numEdges = ImagetoGraphSerial(image, EdgeList, VertexList, BitEdgeList);

    /**** MST Starts ****/
    //Maybe we don't need separate OutList and NWE arrays? Where are we using the NWE array anyway??
    bool DidReduce; //Boolean to check if more segments got created or not

    // This recursive call should return the SuperVertex_Ids which we use to recolor
    // Probably for the hierarchies paper we need to store the SuperVertex_Ids of every iteration

    int32_t tmp_V, tmp_Wt;
    for(int i =0; i<numEdges; i++)
    {
        tmp_V = BitEdgeList[i]% (2 << 15);
        tmp_Wt = BitEdgeList[i]>>16;
        printf("EdgeListV:%d, EdgeListWt:%d, BitVertex:%d, BitWt:%d\n", EdgeList[i].Vertex, EdgeList[i].Weight, tmp_V, tmp_Wt);
    }

    int numthreads = 1024;
    int numBlock = numVertices/numthreads;
    printf("flag: ");
    DidReduce = 1;
    while(DidReduce)
    {
        DidReduce = 0;
        //1. The graph creation step above takes care of this

        //2.
        MarkSegments<<<numBlock, numthreads>>>(flag, VertexList, numVertices);

        //Segmented min scan
        SegmentedReduction(*context, VertexList, BitEdgeList, OutList, NWE, numEdges, numVertices);
        FindSuccessorArray<<<numBlock,numthreads>>>(Successor, NWE, numVertices);

    }

    /*
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
    //11 Removing self edges
    RemoveSelfEdges<<<numBlock,numthreads>>>(BitEdgeList, numEdges, uid, SuperVertexId);

    //12 Remove largest duplicate edges
    int *UV, *W;
    cudaMallocManaged(&UV,numEdges*sizeof(int64_t));
    cudaMallocManaged(&W,numEdges*sizeof(int64_t));
    CreateUVWArray<<<numBlock,numthreads>>>(BitEdgeList, numEdges, uid, SuperVertexId, UV, W);

    int32_t *flag3;
    cudaMallocManaged(&flag3,numEdges*sizeof(int32_t));
    int new_edge_size = SortUVW(UV, W, numEdges, flag3);

    int32_t *compact_locations;
    cudaMallocManaged(&compact_locations,new_edge_size*sizeof(int32_t));
*/
    return 0;
}
