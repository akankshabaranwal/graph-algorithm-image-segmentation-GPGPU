#include <iostream>
#include <cuda_runtime_api.h>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>
#include "CreateGraph.h"
#include "moderngpu.cuh"		// Include all MGPU kernels.
#include "FastMST.h"
#include "RecolorImage.h"

using namespace cv;
using namespace cv::cuda;
using namespace mgpu;
// TODO: Add the error handling code from:
//  http://cuda-programming.blogspot.com/2013/01/vector-addition-in-cuda-cuda-cc-program.html

uint64_t mask_32 = 0x00000000FFFFFFFF;//32 bit mask
uint64_t mask_22 = 0x000003FFFFF;//32 bit mask
uint64_t mask_20 = 0x000000FFFFF;//32 bit mask

int main(int argc, char **argv)
{
    Mat image, output;
    GpuMat dev_image, dev_output;

    image = imread("data/beach.png", IMREAD_COLOR);
    cv::resize(image, image, cv::Size(), 0.05, 0.05);

    printf("Size of image obtained is: Rows: %d, Columns: %d, Pixels: %d\n", image.rows, image.cols, image.rows * image.cols);

    //TODO: Add checker for image size depending on the bits decided for representing edge weight and vertex index
    dev_image.upload(image);

    Ptr<Filter> filter = createGaussianFilter(CV_8UC3, CV_8UC3, Size(5, 5), 1.0);
    filter->apply(dev_image, dev_output);

    //Graph parameters
    int numVertices = image.rows * image.cols;
    uint numEdges = (image.rows) * (image.cols) * 4;

    //Convert image to graph
    uint32_t *VertexList, *MarkedSegments, *OnlyEdge, *OnlyVertex, *FlagList, *NWE, *Successor, *newSuccessor, *L, *Representative, *VertexIds;
    uint64_t *OnlyWeight, *tempArray, *BitEdgeList, *MinSegmentedList;
    uint32_t *MinMaxScanArray;
    uint32_t *new_E_size, *new_V_size;
    uint32_t *compactLocations, *expanded_u;
    uint32_t *C;

    edge *EdgeList;

    uint *flag;
    cudaMallocManaged(&flag, numEdges * sizeof(uint32_t));
    cudaMallocManaged(&MarkedSegments,numEdges * sizeof(uint32_t));
    //Allocating memory
    cudaMallocManaged(&VertexList, numVertices * sizeof(uint32_t));
    cudaMallocManaged(&FlagList, numVertices * sizeof(uint32_t));
    cudaMallocManaged(&MinSegmentedList, numVertices * sizeof(uint64_t));
    cudaMallocManaged(&tempArray, numVertices * sizeof(uint64_t));
    cudaMallocManaged(&EdgeList, numEdges * sizeof(edge));
    cudaMallocManaged(&BitEdgeList, numEdges * sizeof(uint64_t));
    cudaMallocManaged(&NWE, numVertices * sizeof(uint32_t));
    cudaMallocManaged(&Successor, numVertices * sizeof(uint32_t));
    cudaMallocManaged(&newSuccessor, numVertices * sizeof(uint32_t));

    cudaMallocManaged(&OnlyEdge, numEdges * sizeof(uint32_t));
    cudaMallocManaged(&OnlyVertex, numEdges * sizeof(uint32_t));
    cudaMallocManaged(&OnlyWeight, numEdges * sizeof(uint64_t));

    cudaMallocManaged(&L, numVertices * sizeof(uint32_t));
    cudaMallocManaged(&Representative, numVertices * sizeof(uint32_t));
    cudaMallocManaged(&VertexIds, numVertices * sizeof(uint32_t));
    cudaMallocManaged(&new_E_size, numEdges * sizeof(uint32_t));
    cudaMallocManaged(&new_V_size, numEdges * sizeof(uint32_t));
    cudaMallocManaged(&MinMaxScanArray, numEdges * sizeof(uint32_t));
    cudaMallocManaged(&compactLocations, numEdges * sizeof(uint32_t));
    cudaMallocManaged(&expanded_u, numEdges * sizeof(uint32_t));
    cudaMallocManaged(&C, numEdges * sizeof(uint32_t));

    uint *Flag2;
    cudaMallocManaged(&Flag2, numEdges * sizeof(uint32_t));
    uint32_t *SuperVertexId;
    cudaMallocManaged(&SuperVertexId, numVertices * sizeof(uint32_t));

    uint *uid;
    cudaMallocManaged(&uid, numVertices * sizeof(uint32_t));
    dim3 threadsPerBlock(1024, 1024);
    uint BlockX = image.rows / threadsPerBlock.x;
    uint BlockY = image.cols / threadsPerBlock.y;
    dim3 numBlocks(BlockX, BlockY);
    cudaDeviceSynchronize(); //FIXME: Need to check where all this synchronize call is needed
    ContextPtr context = CreateCudaDevice(argc, argv, true);
    cudaError_t err = cudaGetLastError();

    uint *flag4; //Same as F4. New flag for creating vertex list. Assigning the new ids.
    cudaMallocManaged(&flag4,numEdges * sizeof(uint));

    bool *change;
    cudaMallocManaged(&change, sizeof(bool));

    dev_output.download(output);

    uint32_t tmp_V;
    uint64_t tmp_Wt;

    uint numthreads;
    uint numBlock;

    uint64_t *UV, *UVW;
    uint32_t *W;

    cudaMallocManaged(&UV,numEdges*sizeof(uint64_t));
    cudaMallocManaged(&UVW,numEdges*sizeof(uint64_t));
    cudaMallocManaged(&W,numEdges*sizeof(uint32_t));

    uint *flag3;
    cudaMallocManaged(&flag3,numEdges*sizeof(uint));
    uint *Flag4;
    cudaMallocManaged(&Flag4,numEdges*sizeof(uint));

    numEdges = ImagetoGraphSerial(image, EdgeList, VertexList, BitEdgeList);

    /**** MST Starts ****/
    bool DidReduce; //Boolean to check if more segments got created or not
    DidReduce = 1;

   printf("Vertex\n");
    for (uint32_t i = 0; i < numVertices; i++)
    {
        tmp_V = VertexList[i];
        printf("%d, ", tmp_V);
        OnlyVertex[i]=tmp_V;
    }

    printf("\nEdge\n");
    for (uint32_t i = 0; i < numEdges; i++)
    {
        tmp_V = BitEdgeList[i] & mask_32;
        tmp_Wt = BitEdgeList[i]>>32;
        printf("%d %d %d %d, ", tmp_V, tmp_Wt, EdgeList[i].Vertex, EdgeList[i].Weight);
        OnlyEdge[i] = tmp_V;
        OnlyWeight[i] = (tmp_Wt<<32) | i;

        if (tmp_V != EdgeList[i].Vertex)
        {    printf("ERROR!!!");
            exit(-1);
        }
        if (tmp_Wt != EdgeList[i].Weight)
        {    printf("ERROR!!!");
            exit(-1);
        }
    }

    std::vector<uint32_t*> d_hierarchy_levels;	// Vector containing pointers to all hierarchy levels (don't dereference on CPU, device pointers)
    std::vector<int> hierarchy_level_sizes;			// Size of each hierarchy level

    while(numVertices>1)
    {
        printf("***********************************");
        printf("\nStarting new iteration with numVertices=%d, numEdges=%d\n", numVertices, numEdges);
        if(numVertices>1024)
        numthreads = min(1024,numVertices);
        else if(numVertices>512)
            numthreads = min(512,numVertices);
        else if(numVertices>256)
            numthreads = min(256,numVertices);
        else if(numVertices>128)
            numthreads = min(128,numVertices);
        else if(numVertices>64)
            numthreads = min(64,numVertices);
        else
            numthreads = min(32,numVertices);

        numBlock = numVertices/numthreads;

        //1. The graph creation step above takes care of this
        for (uint32_t i = 0; i < numEdges; i++)
        {
            tmp_Wt = BitEdgeList[i] >> 32;
            OnlyWeight[i] = (tmp_Wt << 32) | i;
        }
        //2. Mark the segments in the flag array. Being used for the uid array below
        printf("\nVertices are:\n");
        for(int i=0;i<numVertices; i++)
            printf("%d, ", OnlyVertex[i]);

        printf("\nEdges are:\n");
        for(int i=0;i<numEdges; i++)
            printf("%d, ", OnlyEdge[i]);

        printf("\nWeights are:\n");
        for(int i=0;i<numEdges; i++)
            printf("%d, ", OnlyWeight[i]>>32);

        /*printf("\nPrinting new BitEdgeIndex List:\n");
        for(int i=0;i<numEdges;i++)
            printf("%d, ", BitEdgeList[i]&mask_32);

        printf("\nPrinting new BitEdgeWeight List:\n");
        for(int i=0;i<numEdges;i++)
            printf("%d, ", BitEdgeList[i]>>32);*/

        ClearFlagArray<<<numBlock, numthreads>>>(flag, numEdges);

        cudaError_t err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: Flag Array%s\n", cudaGetErrorString(err));
            exit(-1);
        }

        MarkSegments<<<numBlock, numthreads>>>(flag, VertexList, numEdges);
        cudaDeviceSynchronize();
        //Create UID array. 10.2
        CreateUid(MarkedSegments, flag, numEdges); //Isnt this same as the vertex list??

        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: Mark Segments%s\n", cudaGetErrorString(err));
            exit(-1);
        }
        //IncrementVertexList<<<numBlock, numthreads>>>(VertexList,numVertices);
        //cudaDeviceSynchronize();

        //3. Segmented min scan
        //SegmentedReduction(*context, VertexList, BitEdgeList, MinSegmentedList, numEdges, numVertices);
        SegmentedReduction(*context, VertexList, OnlyWeight, tempArray, numEdges, numVertices);
        //DecrementVertexList<<<numBlock, numthreads>>>(VertexList,numVertices);
        //cudaDeviceSynchronize();

        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: Segment Reduction%s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();
        /*printf("\nOnlyWeightArray is:\n");
        for(int i=0;i<numVertices;i++)
            printf("%d %d %lld,  ", OnlyWeight[i]>>32, OnlyWeight[i]&mask_32, OnlyWeight[i]);*/

        /*printf("\nOnlyWeightArray without bit manipulation is:\n");
        for(int i=0;i<numEdges;i++)
            printf("%lld,  ", OnlyWeight[i]);*/

        //Debug NWE array creation
        /*printf("\nInput to create NWE Array is:\n");
        for(int i=0;i<numVertices;i++)
            printf("%d %d %lld,  ", tempArray[i]>>32, tempArray[i]&mask_32,  tempArray[i]);*/


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
/*
        printf("\nAfter sort BitEdgelist Edges, Vertex are:\n");
        for(int i=0;i<numEdges; i++)
            printf("%d,%d ", BitEdgeList[i]&mask_32,  BitEdgeList[i]>>32);

        printf("\nAfter sort OnlyEdgeList Edges, Vertex are:\n");
        for(int i=0;i<numEdges; i++)
            printf("%d,%d ", OnlyWeight[i]&mask_32,  OnlyWeight[i]>>32);*/

        //4. Find Successor array of each vertex
        FindSuccessorArray<<<numBlock, numthreads>>>(Successor, BitEdgeList, NWE, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: FindSuccessorArray %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();

/*        printf("\nPrinting Successor Array: \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", Successor[i]);
        }*/
        RemoveCycles<<<numBlock, numthreads>>>(Successor, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error RemoveCycles: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();
/*
        printf("\n After removing cycles printing Successor Array: \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", Successor[i]);
        }*/

        //C. Merging vertices and assigning IDs to supervertices
        //7. Propagate representative vertex IDs using pointer doubling
        cudaDeviceSynchronize(); //because PropagateRepresentative is on host

        PropagateRepresentativeVertices(Successor, numVertices);

        cudaDeviceSynchronize();
        printf("\n After propagating representative vertices printing Successor Array: \n");
        for(int i =0; i< numVertices; i++)
        {
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
/*
        printf("\n Representative array, Vertex Array \n");
        for(int i =0; i< numVertices; i++)
        {

            printf("%d %d %d,  ", Successor[i], Representative[i], VertexIds[i]);
        }
*/
        //9. Create F2, Assign new IDs based on Flag2 array
        cudaDeviceSynchronize();

        thrust::sort_by_key(thrust::device, Representative, Representative + numVertices, VertexIds);
        cudaDeviceSynchronize();

        CreateFlag2Array<<<numBlock, numthreads>>>(Representative, Flag2, numVertices);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(Flag2, Flag2 + numVertices, C, thrust::plus<int>());
        cudaDeviceSynchronize();

  /*      printf("\n Sorted representative array, Vertex Array \n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d %d,  ", Representative[i], VertexIds[i]);
        }


        printf("\nFlag2\n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", Flag2[i]);
        }
        printf("\nNew Indices C array is:\n");
        for(int i =0; i< numVertices; i++)
        {
            printf("%d ,", C[i]);
        }*/

        //D. Finding the Supervertex ids and storing it in an array
        CreateSuperVertexArray<<<numBlock,numthreads>>>(SuperVertexId, VertexIds, C, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error CreateSuperVertexArray: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();


        //Create UID array. 10.2
        CreateUid(uid, flag, numEdges); //Isnt this same as the vertex list??
        cudaDeviceSynchronize();

        printf("\n Uid\n");
        for(int i =0; i< numEdges; i++)
        {
            printf("%d ,", uid[i]);
        }
        /*printf("\nPrinting Only Edge Array before self edges: \n");
        for (int i = 0; i < numEdges; i++)
        {
            printf("%d, ", OnlyEdge[i]);
        }*/
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
        /*printf("\nPrinting only Edge Array after marked for removal: \n");
        for (int i = 0; i < numEdges; i++)
        {
            printf("%d, ", OnlyEdge[i]);
        }*/
        //E 12.
        CreateUVWArray<<<numBlock,numthreads>>>(BitEdgeList, OnlyEdge, numEdges, uid, SuperVertexId, UV, W, UVW);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error CreateUVWArray: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();
      /*  printf("\n Printing UVW array: before calling SortUVW");
        for(int i = 0; i< numEdges;i++)
        {
            printf("%d %d %d , ", UV[i]>>32, UV[i]&mask_32, W[i]);
        }*/
        printf("\n");
        //12.2 Sort the UVW Array
        thrust::sort_by_key(thrust::device, UV, UV + numEdges, W);
        thrust::sort_by_key(thrust::device, UVW, UVW + numEdges, W);

        cudaDeviceSynchronize();
        /*printf("\n Printing UVW array after SortUVW: ");
        for(int i = 0; i< numEdges;i++)
        {
            printf("%d %d %d , ", UV[i]>>32, UV[i]&mask_32, W[i]);
        }

        printf("\n Printing bitUVW array after SortUVW: ");
        for(int i = 0; i< numEdges;i++)
        {
            printf("%d %d %d , ", UVW[i]>>44, ((UVW[i]>>22)&mask_22),  (UVW[i]&mask_20));
        }
*/
        printf("\n");
        flag3[0]=1;
        CreateFlag3Array<<<numBlock,numthreads>>>(UV, W, numEdges, flag3, MinMaxScanArray);
        cudaDeviceSynchronize();
        printf("\n Printing MinMaxScanArray UV W\n");
       /* for(int i = 0; i< numEdges;i++)
        {
            printf("%d %d %d, ", MinMaxScanArray[i], UV[i]>>32, UV[i]&mask_32);
        }
        printf("\n");*/
        uint32_t *new_edge_size = thrust::max_element(thrust::device, MinMaxScanArray, MinMaxScanArray + numEdges);
        cudaDeviceSynchronize();
        *new_edge_size = *new_edge_size+1;
        printf("\nnew_edge_size %d", *new_edge_size);

        thrust::inclusive_scan(flag3, flag3 + *new_edge_size, compactLocations, thrust::plus<int>());
        cudaDeviceSynchronize();
      /*  printf("\n Printing compact locations array before subtract\n");
        for(int i = 0; i< *new_edge_size;i++)
        {
            printf("%d, ", compactLocations[i]);
        }
        printf("\n");*/
        ResetCompactLocationsArray<<<numBlock,numthreads>>>(compactLocations, *new_edge_size);
        cudaDeviceSynchronize();
/*

        printf("\nPrinting inputs for CreatNewEdgeList\n");
        printf("\n Printing flag3 array\n");
        for(int i = 0; i< *new_edge_size;i++)
        {
            printf("%d, ", flag3[i]);
        }

        printf("\n Printing compact locations array\n");
        for(int i = 0; i< *new_edge_size;i++)
        {
            printf("%d, ", compactLocations[i]);
        }
        printf("\n");
        printf("\n Printing UVW array\n");

        for(int i=0; i< *new_edge_size; i++)
        {
            printf("%d %d %d, ", UV[i]>>32, UV[i]&mask_32, W[i]);
        }
        printf("\n");*/
        CreateNewEdgeList<<<numBlock,numthreads>>>( BitEdgeList, compactLocations, OnlyEdge, OnlyWeight, UV, W, UVW, flag3, *new_edge_size, new_E_size, new_V_size, expanded_u);

        uint32_t *new_E_sizeptr = thrust::max_element(thrust::device, new_E_size, new_E_size + *new_edge_size);
        uint32_t *new_V_sizeptr = thrust::max_element(thrust::device, new_V_size, new_V_size + *new_edge_size);
        cudaDeviceSynchronize();
        numVertices = *new_V_sizeptr;
        numEdges = *new_E_sizeptr;
        cudaDeviceSynchronize();

/*        printf("\nAfter CreateNewEdgeList\n");

        printf("\nPrinting E\n");
        for(int i=0; i< numEdges;i++)
            printf("%d, ", OnlyEdge[i]);

        printf("\nPrinting W\n");
        for(int i=0; i< numEdges;i++)
            printf("%d, ", OnlyWeight[i]>>32);

        printf("\nPrinting expanded_u\n");
        for(int i=0; i< numEdges;i++)
            printf("%d, ", expanded_u[i]);*/

        Flag4[0]=1;
        CreateFlag4Array<<<numBlock,numthreads>>>(expanded_u, Flag4, numEdges);
        cudaDeviceSynchronize();

    /*    printf("\nPrinting expanded_u\n");
        for(int i=0; i<numEdges; i++)
                printf("%d, ", expanded_u[i]);

        printf("\nPrinting Flag4\n");
        for(int i=0; i<numEdges; i++)
            printf("%d, ", Flag4[i]);*/

        CreateNewVertexList<<<numBlock,numthreads>>>(OnlyVertex, VertexList, Flag4, numEdges, expanded_u);

        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: CreateNewVertexList%s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();

     /* printf("\n numVertices: %d numEdges %d", numVertices, numEdges);

        printf("\nPrinting new Vertex List:\n");
        for(int i=0;i<numVertices;i++)
            printf("%d, ", OnlyVertex[i]);

        printf("\nPrinting new Edge List:\n");
        for(int i=0;i<numEdges;i++)
            printf("%d, ", OnlyEdge[i]);

        printf("\nPrinting new Weight List:\n");
        for(int i=0;i<numEdges;i++)
            printf("%d, ", OnlyWeight[i]>>32);

        printf("\nPrinting new BitEdgeIndex List:\n");
        for(int i=0;i<numEdges;i++)
            printf("%d, ", BitEdgeList[i]&mask_32);

        printf("\nPrinting new BitEdgeWeight List:\n");
        for(int i=0;i<numEdges;i++)
            printf("%d, ", BitEdgeList[i]>>32);*/

        d_hierarchy_levels.push_back(SuperVertexId);
        hierarchy_level_sizes.push_back(numVertices);
      cudaMallocManaged(&SuperVertexId, numVertices * sizeof(int32_t));
        //numVertices=1;
      //return 0;
    }
    std::string outFile="test";
    writeComponents(d_hierarchy_levels, image.rows*image.cols, 3, hierarchy_level_sizes, outFile, image.rows, image.cols);
    return 0;
}
