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
    int scale_percent = 10; // percent of original size
    cv::resize(image, image, cv::Size(), 0.05, 0.05);

    printf("Size of image obtained is: Rows: %d, Columns: %d, Pixels: %d\n", image.rows, image.cols, image.rows * image.cols);

    //TODO: Add checker for image size depending on the bits decided for representing edge weight and vertex index
    dev_image.upload(image);

    Ptr<Filter> filter = createGaussianFilter(CV_8UC3, CV_8UC3, Size(5, 5), 1.0);
    filter->apply(dev_image, dev_output);

    //Graph parameters
    int numVertices = image.rows * image.cols;
    int numEdges = (image.rows) * (image.cols) * 4;

    //Convert image to graph
    int32_t *VertexList, *OnlyEdge, *OnlyVertex, *OnlyWeight, *BitEdgeList, *FlagList, *MinSegmentedList, *tempArray, *NWE, *Successor, *newSuccessor, *L, *Representative, *VertexIds;
    int32_t *MinMaxScanArray;
    int32_t *new_E_size, *new_V_size;
    int32_t *compactLocations, *expanded_u;

    edge *EdgeList;

    int *flag;
    cudaMallocManaged(&flag, numEdges * sizeof(int32_t));
    //Allocating memory
    cudaMallocManaged(&VertexList, numVertices * sizeof(int32_t));
    cudaMallocManaged(&FlagList, numVertices * sizeof(int32_t));
    cudaMallocManaged(&MinSegmentedList, numVertices * sizeof(int32_t));
    cudaMallocManaged(&tempArray, numVertices * sizeof(int32_t));
    cudaMallocManaged(&EdgeList, numEdges * sizeof(edge));
    cudaMallocManaged(&BitEdgeList, numEdges * sizeof(int32_t));
    cudaMallocManaged(&NWE, numVertices * sizeof(int32_t));
    cudaMallocManaged(&Successor, numVertices * sizeof(int32_t));
    cudaMallocManaged(&newSuccessor, numVertices * sizeof(int32_t));

    cudaMallocManaged(&OnlyEdge, numEdges * sizeof(int32_t));
    cudaMallocManaged(&OnlyVertex, numEdges * sizeof(int32_t));
    cudaMallocManaged(&OnlyWeight, numEdges * sizeof(int32_t));

    cudaMallocManaged(&L, numVertices * sizeof(int32_t));
    cudaMallocManaged(&Representative, numVertices * sizeof(int32_t));
    cudaMallocManaged(&VertexIds, numVertices * sizeof(int32_t));
    cudaMallocManaged(&new_E_size, numEdges * sizeof(int32_t));
    cudaMallocManaged(&new_V_size, numEdges * sizeof(int32_t));
    cudaMallocManaged(&MinMaxScanArray, numEdges * sizeof(int32_t));
    cudaMallocManaged(&compactLocations, numEdges * sizeof(int32_t));
    cudaMallocManaged(&expanded_u, numEdges * sizeof(int32_t));


    int *Flag2;
    cudaMallocManaged(&Flag2, numEdges * sizeof(int32_t));
    int *SuperVertexId;
    cudaMallocManaged(&SuperVertexId, numVertices * sizeof(int32_t));

    int *uid;
    cudaMallocManaged(&uid, numVertices * sizeof(int32_t));
    dim3 threadsPerBlock(32, 32);
    int BlockX = image.rows / threadsPerBlock.x;
    int BlockY = image.cols / threadsPerBlock.y;
    dim3 numBlocks(BlockX, BlockY);
    cudaDeviceSynchronize(); //FIXME: Need to check where all this synchronize call is needed
    ContextPtr context = CreateCudaDevice(argc, argv, true);
    cudaError_t err = cudaGetLastError();

    int *flag4; //Same as F4. New flag for creating vertex list. Assigning the new ids.
    cudaMallocManaged(&flag4, numEdges * sizeof(int));

    bool *change;
    cudaMallocManaged(&change, sizeof(bool));
    //FIXME: Make this initialization run in parallel?
    //TODO: Figure out if this initialization is required??
    for (int i = 0; i < numEdges; i++)
    {
        EdgeList[i].Weight = 0;
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

    int numthreads = 32;
    int numBlock = numVertices/numthreads;

    int *UV, *W;
    cudaMallocManaged(&UV,numEdges*sizeof(int64_t));
    cudaMallocManaged(&W,numEdges*sizeof(int64_t));
    int32_t *flag3;
    cudaMallocManaged(&flag3,numEdges*sizeof(int32_t));
    int32_t *Flag4;
    cudaMallocManaged(&Flag4,numEdges*sizeof(int32_t));

    numEdges = ImagetoGraphSerial(image, EdgeList, VertexList, BitEdgeList);

    /**** MST Starts ****/
    bool DidReduce; //Boolean to check if more segments got created or not
    DidReduce = 1;

    printf("Vertex\n");
    for (int i = 0; i < numVertices; i++)
    {
        tmp_V = VertexList[i];
        printf("%d, ", tmp_V);
        OnlyVertex[i]=tmp_V;
    }

    printf("\nEdge\n");
    for (int i = 0; i < numEdges; i++)
    {
        tmp_V = BitEdgeList[i] % (2 << 15);
        printf("%d, ", tmp_V);
        OnlyEdge[i] = tmp_V;
        if (tmp_V != EdgeList[i].Vertex)
        {    printf("ERROR!!!");
            exit(-1);
        }
    }

    printf("\nWeight\n");
    for (int32_t i = 0; i < numEdges; i++)
    {
        tmp_Wt = BitEdgeList[i]>>16;
        printf("%d, ", tmp_Wt);
        OnlyWeight[i]=(tmp_Wt * (2<<15)) + i;
       /* if ((int32_t(tmp_Wt) - int32_t(EdgeList[i].Weight))!=0);
        {   printf("%d ", tmp_Wt - EdgeList[i].Weight);
            printf("ERROR!!!");
            exit(-1);
        }*/
    }
    //return 0;

    while(numVertices>1)
    {
        printf("it\n");

        //1. The graph creation step above takes care of this

        //2. Mark the segments in the flag array. Being used for the uid array below
        ClearFlagArray<<<numBlock, numthreads>>>(flag, numEdges);

        cudaError_t err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: Flag Array%s\n", cudaGetErrorString(err));
            exit(-1);
        }

        MarkSegments<<<numBlock, numthreads>>>(flag, VertexList, numEdges);

        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: Mark Segments%s\n", cudaGetErrorString(err));
            exit(-1);
        }
        printf("\n Printing Only Weight Array Indices before minsegment: \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", OnlyWeight[i]%(2<<15));
        }
        printf("\n Printing BitEdgeList Array before minsegment: \n");
        for(int i =0; i< numEdges; i++)
        {
            printf("%d ,", BitEdgeList[i]);
        }
        //3. Segmented min scan
        SegmentedReduction(*context, VertexList, BitEdgeList, MinSegmentedList, numEdges, numVertices);
        SegmentedReduction(*context, VertexList, OnlyWeight, tempArray, numEdges, numVertices);

        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: Segment Reduction%s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();
        printf("\n Printing MinSegment Array Values: \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", MinSegmentedList[i]);
        }
        printf("\n Printing MinSegment Array Weights: \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", MinSegmentedList[i]>>16);
        }
        printf("\n Printing MinSegment Array Indices: \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", MinSegmentedList[i]%(2<<15));
        }
        printf("\n Printing Only Weight Array Indices: \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", tempArray[i]%(2<<15));
        }
        // Create NWE array
        CreateNWEArray<<<numBlock, numthreads>>>(NWE, tempArray, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: CreateNWEArray %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();
        printf("\n Printing NWE Array: \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", NWE[i]);
        }
        //4. Find Successor array of each vertex
        FindSuccessorArray<<<numBlock, numthreads>>>(Successor, BitEdgeList, NWE, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: FindSuccessorArray %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();

        printf("\nPrinting Successor Array: \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", Successor[i]);
        }

        RemoveCycles<<<numBlock, numthreads>>>(Successor, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error RemoveCycles: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();

        printf("\n After removing cycles printing Successor Array: \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", Successor[i]);
        }

        //C. Merging vertices and assigning IDs to supervertices
        //7. Propagate representative vertex IDs using pointer doubling
        cudaDeviceSynchronize(); //because PropagateRepresentative is on host

        PropagateRepresentativeVertices(Successor, numVertices);

        cudaDeviceSynchronize();
        printf("\n After propagating representative vertices printing Successor Array: \n");
        for(int i =0; i< numVertices; i++)
        {
            if(Successor[i]!=18145)
          printf("%d ,", Successor[i]);
        }

        //8, 9 Append appendSuccessorArray
        appendSuccessorArray<<<numBlock, numthreads>>>(Representative, VertexIds, Successor, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: AppendSuccessorArray %s\n", cudaGetErrorString(err));
            exit(-1);
            exit(-1);
        }
        cudaDeviceSynchronize();

        printf("\n Representative array \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", Representative[i]);
        }

        printf("\n Vertex \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", VertexIds[i]);
        }
        //cudaDeviceSynchronize();
        //9. Create F2, Assign new IDs based on Flag2 array
        cudaDeviceSynchronize();
        SortedSplit(Representative, VertexIds, Successor, Flag2, numVertices);
        cudaDeviceSynchronize();

        printf("\n Sorted representative array \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", Representative[i]);
        }

        printf("\n Sorted Vertex Labels \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", VertexIds[i]);
        }

        printf("Flag\n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", Flag2[i]);
        }

        //D. Finding the Supervertex ids and storing it in an array
        CreateSuperVertexArray<<<numBlock,numthreads>>>(SuperVertexId, VertexIds, Flag2, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error CreateSuperVertexArray: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();
       printf("\n SuperVertexIds\n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", SuperVertexId[i]);
        }

        //Create UID array. 10.2
        CreateUid(uid, flag, numEdges); //Isnt this same as the vertex list??
        cudaDeviceSynchronize();

        printf("\n Uid\n");
        for(int i =0; i< numEdges; i++)
        {
            printf("%d ,", uid[i]);
        }
        printf("\nPrinting Only Edge Array before self edges: \n");
        for (int i = 0; i < numEdges; i++)
        {
            printf("%d, ", OnlyEdge[i]);
        }
        //11. Removing self edges
        RemoveSelfEdges<<<numBlock,numthreads>>>(OnlyEdge, numEdges, uid, SuperVertexId);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error RemoveSelfEdges: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();

        printf("\n SuperVertex\n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", SuperVertexId[i]);
        }
        printf("\nPrinting only Edge Array after marked for removal: \n");
        for (int i = 0; i < numEdges; i++)
        {
            if(OnlyEdge[i]!=INT_MAX)
            printf("%d, ", OnlyEdge[i]);
        }
        //E 12.
        CreateUVWArray<<<numBlock,numthreads>>>(BitEdgeList, OnlyEdge, numEdges, uid, SuperVertexId, UV, W);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error CreateUVWArray: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();
        printf("\n Printing UVW array: before calling SortUVW");
        for(int i = 0; i< numEdges;i++)
        {
            printf("%d %d %d , ", UV[i]>>16, UV[i]%(2<<15), W[i]);
        }
        printf("\n");
        //12.2 Sort the UVW Array
        thrust::sort_by_key(thrust::device, UV, UV + numEdges, W);
        cudaDeviceSynchronize();
        printf("\n Printing UVW array after SortUVW: ");
        for(int i = 0; i< numEdges;i++)
        {
            printf("%d %d %d , ", UV[i]>>16, UV[i]%(2<<15), W[i]);
        }
        printf("\n");
        flag3[0]=0;
        CreateFlag3Array<<<numBlock,numthreads>>>(UV, W, numEdges, flag3, MinMaxScanArray);

        int *new_edge_size = thrust::max_element(thrust::device, MinMaxScanArray, MinMaxScanArray + numEdges);
        cudaDeviceSynchronize();
        printf("new_edge_size %d", *new_edge_size);
        thrust::inclusive_scan(flag3, flag3 + *new_edge_size, compactLocations, thrust::plus<int>());
        cudaDeviceSynchronize();

        printf("Printing flag3 array\n");
        for(int i = 0; i< numEdges;i++)
        {
            printf("%d, ", flag3[i]);
        }
        printf("\n");
        CreateNewEdgeList<<<numBlock,numthreads>>>( BitEdgeList, compactLocations, OnlyEdge, OnlyWeight, UV, W, flag3, *new_edge_size, new_E_size, new_V_size, expanded_u);
        int *new_E_sizeptr = thrust::max_element(thrust::device, new_E_size, new_E_size + numEdges);
        int *new_V_sizeptr = thrust::max_element(thrust::device, new_V_size, new_V_size + numEdges);
        cudaDeviceSynchronize();
        numVertices = *new_V_sizeptr;
        numEdges = *new_E_sizeptr;
        printf("%d, %d ", *new_E_sizeptr, *new_V_sizeptr);
        CreateFlag4Array<<<numBlock,numthreads>>>(expanded_u, Flag4, *new_E_size);
        CreateNewVertexList<<<numBlock,numthreads>>>(OnlyVertex, Flag4, *new_E_size, expanded_u);

        //12.3 Create Flag3 array. This gives the new_edge_size variable

        //13.1 Create CompactLocations array based on Flag3 above. Scan

        //13.2 Create expanded_u array


        /*
        //flag3 could be renamed to compact location
        numVertices = CreateNewEdgeVertexList(BitEdgeList, VertexList, UV, W, flag3, new_edge_size, flag4);

        //numEdges = new_edge_size; //This is incorrect. Need to return new_E_size as well*/
        cudaDeviceSynchronize();

        printf("\n numVertices: %d numEdges %d", numVertices, numEdges);

        //numVertices = 1;
    }

    return 0;
}
