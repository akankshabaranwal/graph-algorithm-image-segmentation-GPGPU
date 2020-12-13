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
// TODO: Add the error handling code from:
//  http://cuda-programming.blogspot.com/2013/01/vector-addition-in-cuda-cuda-cc-program.html
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
    int32_t *VertexList, *BitEdgeList, *FlagList, *OutList, *NWE, *Successor, *newSuccessor, *L, *Representative, *VertexIds;
    edge *EdgeList;

    int *flag;
    cudaMallocManaged(&flag,numEdges*sizeof(int32_t));
    //Allocating memory
    cudaMallocManaged(&VertexList,numVertices*sizeof(int32_t));
    cudaMallocManaged(&FlagList,numVertices*sizeof(int32_t));
    cudaMallocManaged(&OutList,numVertices*sizeof(int32_t));
    cudaMallocManaged(&EdgeList,numEdges*sizeof(edge));
    cudaMallocManaged(&BitEdgeList,numEdges*sizeof(int32_t));
    cudaMallocManaged(&NWE,numVertices*sizeof(int32_t));
    cudaMallocManaged(&Successor,numVertices*sizeof(int32_t));
    cudaMallocManaged(&newSuccessor,numVertices*sizeof(int32_t));

    cudaMallocManaged(&L,numVertices*sizeof(int32_t));
    cudaMallocManaged(&Representative,numVertices*sizeof(int32_t));
    cudaMallocManaged(&VertexIds,numVertices*sizeof(int32_t));

    int *Flag2;
    cudaMallocManaged(&Flag2,numEdges*sizeof(int32_t));
    int *SuperVertexId;
    cudaMallocManaged(&SuperVertexId,numVertices*sizeof(int32_t));

    int *uid;
    cudaMallocManaged(&uid,numVertices*sizeof(int32_t));
    dim3 threadsPerBlock(32,32);
    int BlockX = image.rows/threadsPerBlock.x;
    int BlockY = image.cols/threadsPerBlock.y;
    dim3 numBlocks(BlockX, BlockY);
    cudaDeviceSynchronize();//FIXME: Need to check where all this synchronize call is needed
    ContextPtr context = CreateCudaDevice(argc, argv, true);
    cudaError_t err = cudaGetLastError();

    int *flag4; //Same as F4. New flag for creating vertex list. Assigning the new ids.
    cudaMallocManaged(&flag4, numEdges * sizeof(int));

    bool *change;
    cudaMallocManaged(&change, sizeof(bool));
    //FIXME: Make this initialization run in parallel?
    //TODO: Figure out if this initialization is required??
    for(int i =0;i<numEdges;i++)
    {
    EdgeList[i].Weight=0;
    }

    dev_output.download(output);

    int32_t tmp_V, tmp_Wt;
    /*
    for(int i =0; i<numEdges; i++)
    {
        tmp_V = BitEdgeList[i]% (2 << 15);
        tmp_Wt = BitEdgeList[i]>>16;
        printf("EdgeListV:%d, EdgeListWt:%d, BitVertex:%d, BitWt:%d\n", EdgeList[i].Vertex, EdgeList[i].Weight, tmp_V, tmp_Wt);
    }*/

    int numthreads = 1024;
    int numBlock = numVertices/numthreads;

    int *UV, *W;
    cudaMallocManaged(&UV,numEdges*sizeof(int64_t));
    cudaMallocManaged(&W,numEdges*sizeof(int64_t));
    int32_t *flag3;
    cudaMallocManaged(&flag3,numEdges*sizeof(int32_t));

    numEdges = ImagetoGraphSerial(image, EdgeList, VertexList, BitEdgeList);

    /**** MST Starts ****/
    bool DidReduce; //Boolean to check if more segments got created or not
    DidReduce = 1;

    while(numVertices>1)
    {
        printf("it\n");

        //1. The graph creation step above takes care of this

        //2. Mark the segments in the flag array. Being used for the uid array below
        ClearFlagArray<<<numBlock, numthreads>>>(flag, numEdges);
        MarkSegments<<<numBlock, numthreads>>>(flag, VertexList, numEdges);

        //3. Segmented min scan
        SegmentedReduction(*context, VertexList, BitEdgeList, OutList, numEdges, numVertices);

        //4. Find Successor array of each vertex
        FindSuccessorArray<<<numBlock, numthreads>>>(Successor, VertexList, OutList, numVertices);
        cudaDeviceSynchronize();

        /*printf("\n Printing Segmented Array: \n");
        for(int i =0; i< 1000; i++)
        {
            printf("%d ,", OutList[i]);
        }
        printf("Printing Successor Array: \n");
        for(int i =0; i< 1000; i++)
        {
            printf("%d ,", Successor[i]);
        }*/

        //5. Remove cycle making edges from NWE. But NWE is not used anywhere here??
        RemoveCycles<<<numBlock, numthreads>>>(Successor, numVertices);
        cudaDeviceSynchronize();
        /*printf("\n After removing cycles printing Successor Array: \n");
        for(int i =0; i< 1000; i++)
        {
            printf("%d ,", Successor[i]);
        }*/

        //C. Merging vertices and assigning IDs to supervertices
        //7. Propagate representative vertex IDs using pointer doubling

        cudaDeviceSynchronize(); //because PropagateRepresentative is on host
        //bool change;
        *change = true;
        //Code copied
        /*do{
            *change=false;
            CopySuccessorToNewSuccessor<<<numBlock, numthreads>>>(Successor, newSuccessor, numVertices);
            cudaDeviceSynchronize();
            PropagateRepresentativeVertices<<<numBlock, numthreads>>>(Successor, newSuccessor, numVertices, change);
            cudaDeviceSynchronize();
            CopyNewSuccessorToSuccessor<<<numBlock, numthreads>>>(Successor, newSuccessor, numVertices);
            cudaDeviceSynchronize();
        }while(change);*/

        cudaDeviceSynchronize();
        printf("\n After propagating representative vertices printing Successor Array: \n");
        for(int i =0; i< 1000; i++)
        {
            printf("%d ,", Successor[i]);
        }
        printf("\n After propagating representative vertices printing Successor Array: \n");

        for(int i =59000; i< 60000; i++)
        {
            printf("%d ,", Successor[i]);
        }

        //8, 9 Append appendSuccessorArray
        appendSuccessorArray<<<numBlock, numthreads>>>(Representative, VertexIds, Successor, numVertices);
        cudaDeviceSynchronize();

        printf("\n Representative array \n");
        for(int i =0; i< 1000; i++)
        {
            printf("%d ,", Representative[i]);
        }

        printf("\n Representative array \n");
        for(int i =59000; i< 60000; i++)
        {
            printf("%d ,", Representative[i]);
        }

        printf("\n Vertex \n");
        for(int i =0; i< 1000; i++)
        {
            printf("%d ,", VertexIds[i]);
        }

        //cudaDeviceSynchronize();
        //9. Create F2, Assign new IDs based on Flag2 array
        cudaDeviceSynchronize();
        //thrust::sort_by_key(thrust::host, VertexIds, VertexIds + numVertices, Representative);
        //thrust::sort_by_key(thrust::host, Representative, Representative + numVertices, VertexIds);
        SortedSplit(Representative, VertexIds, Successor, Flag2, numVertices);
        cudaDeviceSynchronize();

        printf("\n Sorted representative array \n");
        for(int i =0; i< 1000; i++)
        {
            printf("%d ,", Representative[i]);
        }

        printf("\n Sorted Representative array \n");
        for(int i =59000; i< 60000; i++)
        {
            printf("%d ,", Representative[i]);
        }

        printf("\n Sorted Vertex Labels \n");
        for(int i =0; i< 1000; i++)
        {
            printf("%d ,", VertexIds[i]);
        }

        printf("Flag\n");
        for(int i =0; i< 2000; i++)
        {
            printf("%d ,", Flag2[i]);
        }
        printf("Flag\n");
        for(int i =58000; i< 60000; i++)
        {
            printf("%d ,", Flag2[i]);
        }

        //D. Finding the Supervertex ids and storing it in an array
        CreateSuperVertexArray<<<numBlock,numthreads>>>(SuperVertexId, VertexIds, Flag2, numVertices);
        cudaDeviceSynchronize();
        printf("\n SuperVertexIds\n");
        for(int i =0; i< 2000; i++)
        {
            printf("%d ,", SuperVertexId[i]);
        }

        printf("\n SuperVertexId\n");
        for(int i =58000; i< 60000; i++)
        {
            printf("%d ,", SuperVertexId[i]);
        }

        //Create UID array. 10.2
        CreateUid(uid, flag, numVertices); //Isnt this same as the vertex list??
        cudaDeviceSynchronize();

        printf("\n Uid\n");
        for(int i =0; i< 2000; i++)
        {
            printf("%d ,", uid[i]);
        }

        printf("\n Uid\n");
        for(int i =58000; i< 60000; i++)
        {
            printf("%d ,", uid[i]);
        }

        //11. Removing self edges
        RemoveSelfEdges<<<numBlock,numthreads>>>(BitEdgeList,numEdges, uid, SuperVertexId);
        cudaDeviceSynchronize();

        printf("\n SuperVertex\n");
        for(int i =0; i< 2000; i++)
        {
            printf("%d ,", SuperVertexId[i]);
        }

        printf("\n SuperVertex\n");
        for(int i =58000; i< 60000; i++)
        {
            printf("%d ,", SuperVertexId[i]);
        }

        //E 12.
        CreateUVWArray<<<numBlock,numthreads>>>(BitEdgeList, numEdges, uid, SuperVertexId, UV, W);
        cudaDeviceSynchronize();
        printf("\n Printing UVW array: ");
        for(int i = 0; i< 1000;i++)
        {
            printf("%d, ", UV[i]);
        }
        printf("\n");
        int new_edge_size = SortUVW(UV, W, numEdges, flag3);
        cudaDeviceSynchronize();
        printf("\n Printing UVW array: ");
        for(int i = 0; i< 1000;i++)
        {
            printf("%d, ", UV[i]);
        }
        printf("\n");
        printf("\n Printing UVW array: ");
        for(int i = 59000; i< 60000;i++)
        {
            printf("%d, ", UV[i]);
        }/*
        //flag3 could be renamed to compact location
        //numVertices = CreateNewEdgeVertexList(BitEdgeList, VertexList, UV, W, flag3, new_edge_size, flag4);
        //numEdges = new_edge_size; //This is incorrect. Need to return new_E_size as well
        //printf("\n numVertices: %d numEdges %d", numVertices, numEdges);*/
        numVertices = 1;

    }

    return 0;
}
