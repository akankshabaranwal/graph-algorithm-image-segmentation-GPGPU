#include <iostream>
#include <cuda_runtime_api.h>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>
#include "CreateGraph.h"
#include "moderngpu.cuh"		// Include all MGPU kernels.
#include "FastMST.h"
#include "RecolorImage.h"

// Command line options
#include <getopt.h>
#include "Options.h"

using namespace cv;
using namespace cv::cuda;
using namespace mgpu;
// TODO: Add the error handling code from:
//  http://cuda-programming.blogspot.com/2013/01/vector-addition-in-cuda-cuda-cc-program.html

uint64_t mask_32 = 0x00000000FFFFFFFF;//32 bit mask
uint64_t mask_22 = 0x000003FFFFF;//32 bit mask
uint64_t mask_20 = 0x000000FFFFF;//32 bit mask

void segment(Mat image)
{
    Mat output;

    int numVertices = image.rows * image.cols;
    uint numEdges = (image.rows) * (image.cols) * 4;

    //Convert image to graph
    uint32_t *VertexList, *OnlyEdge, *FlagList, *NWE, *Successor, *newSuccessor, *L, *Representative, *VertexIds;
    uint64_t *OnlyWeight, *tempArray, *BitEdgeList, *MinSegmentedList;
    uint32_t *MinMaxScanArray;
    uint32_t *new_E_size, *new_V_size;
    uint32_t *compactLocations, *expanded_u;
    uint32_t *C;
    edge *EdgeList;
    uint *flag;
    uint *Flag2;
    uint32_t *SuperVertexId;
    uint *uid;
    uint *flag4;
    uint64_t *UV, *UVW;
    uint32_t *W;

    uint *flag3;
    uint *Flag4;

    //Allocating memory
    cudaMallocManaged(&flag, numEdges * sizeof(uint32_t));
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
    cudaMallocManaged(&Flag2, numEdges * sizeof(uint32_t));
    cudaMallocManaged(&SuperVertexId, numVertices * sizeof(uint32_t));
    cudaMallocManaged(&uid, numVertices * sizeof(uint32_t));
    cudaMallocManaged(&flag4,numEdges * sizeof(uint));
    cudaMallocManaged(&UV,numEdges*sizeof(uint64_t));
    cudaMallocManaged(&UVW,numEdges*sizeof(uint64_t));
    cudaMallocManaged(&W,numEdges*sizeof(uint32_t));
    cudaMallocManaged(&flag3,numEdges*sizeof(uint));
    cudaMallocManaged(&Flag4,numEdges*sizeof(uint));

    dim3 threadsPerBlock(1024, 1024);
    uint BlockX = image.rows / threadsPerBlock.x;
    uint BlockY = image.cols / threadsPerBlock.y;

    dim3 numBlocks(BlockX, BlockY);
    ContextPtr context = CreateCudaDevice(0);//If CUDA Device error then fix this
    cudaError_t err = cudaGetLastError();

    uint numthreads;
    uint numBlock;

    ImagetoGraphParallelStream(image, VertexList, OnlyEdge, OnlyWeight);

    numEdges = 	8 + 6 * (image.cols - 2) + 6 * (image.rows - 2) + 4 * (image.cols - 2) * (image.rows - 2);

    std::vector<uint32_t*> d_hierarchy_levels;	// Vector containing pointers to all hierarchy levels (don't dereference on CPU, device pointers)
    std::vector<int> hierarchy_level_sizes;			// Size of each hierarchy level
    numthreads = min(32, numVertices);
    numBlock = numVertices/numthreads;
    SetBitEdgeListArray<<<numBlock, numthreads>>>(BitEdgeList, OnlyEdge, OnlyWeight, numEdges);
    cudaDeviceSynchronize();

    while(numVertices>1)
    {
        if(numVertices>1024)
            numthreads = 1024;
        else if(numVertices>512)
            numthreads = 512;
        else if(numVertices>256)
            numthreads = 256;
        else if(numVertices>128)
            numthreads = 128;
        else if(numVertices>64)
            numthreads = 64;
        else
            numthreads = min(32, numVertices);

        numBlock = numVertices/numthreads;

        //1. The graph creation step above takes care of this
        SetOnlyWeightArray<<<numBlock, numthreads>>>(BitEdgeList, OnlyWeight, numEdges);
        cudaError_t err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: SetOnlyWeightArray%s\n", cudaGetErrorString(err));
            exit(-1);
        }

        ClearFlagArray<<<numBlock, numthreads>>>(flag, numEdges);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: Flag Array%s\n", cudaGetErrorString(err));
            exit(-1);
        }

        MarkSegments<<<numBlock, numthreads>>>(flag, VertexList, numEdges);
        //3. Segmented min scan
        SegmentedReduction(*context, VertexList, OnlyWeight, tempArray, numEdges, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: Segment Reduction%s\n", cudaGetErrorString(err));
            exit(-1);
        }
        // Create NWE array
        CreateNWEArray<<<numBlock, numthreads>>>(NWE, tempArray, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: CreateNWEArray %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        //4. Find Successor array of each vertex
        FindSuccessorArray<<<numBlock, numthreads>>>(Successor, BitEdgeList, NWE, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: FindSuccessorArray %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        RemoveCycles<<<numBlock, numthreads>>>(Successor, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error RemoveCycles: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();

        //C. Merging vertices and assigning IDs to supervertices
        //7. Propagate representative vertex IDs using pointer doubling

        PropagateRepresentativeVertices(Successor, numVertices);

        //8, 9 Append appendSuccessorArray
        appendSuccessorArray<<<numBlock, numthreads>>>(Representative, VertexIds, Successor, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: AppendSuccessorArray %s\n", cudaGetErrorString(err));
            exit(-1);
            exit(-1);
        }
        thrust::sort_by_key(thrust::device, Representative, Representative + numVertices, VertexIds);

        CreateFlag2Array<<<numBlock, numthreads>>>(Representative, Flag2, numVertices);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(Flag2, Flag2 + numVertices, C, thrust::plus<int>());

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
        CreateUid(uid, flag, numEdges);

        //11. Removing self edges
        RemoveSelfEdges<<<numBlock,numthreads>>>(OnlyEdge, numEdges, uid, SuperVertexId);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error RemoveSelfEdges: %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        //E 12.
        CreateUVWArray<<<numBlock,numthreads>>>(BitEdgeList, OnlyEdge, numEdges, uid, SuperVertexId, UV, W, UVW);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error CreateUVWArray: %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        //12.2 Sort the UVW Array
        thrust::sort_by_key(thrust::device, UV, UV + numEdges, W);
        thrust::sort_by_key(thrust::device, UVW, UVW + numEdges, W);

        flag3[0]=1;
        CreateFlag3Array<<<numBlock,numthreads>>>(UV, W, numEdges, flag3, MinMaxScanArray);

        uint32_t *new_edge_size = thrust::max_element(thrust::device, MinMaxScanArray, MinMaxScanArray + numEdges);
        cudaDeviceSynchronize();
        *new_edge_size = *new_edge_size+1;
        thrust::inclusive_scan(flag3, flag3 + *new_edge_size, compactLocations, thrust::plus<int>());

        ResetCompactLocationsArray<<<numBlock,numthreads>>>(compactLocations, *new_edge_size);
        CreateNewEdgeList<<<numBlock,numthreads>>>( BitEdgeList, compactLocations, OnlyEdge, OnlyWeight, UV, W, UVW, flag3, *new_edge_size, new_E_size, new_V_size, expanded_u);

        uint32_t *new_E_sizeptr = thrust::max_element(thrust::device, new_E_size, new_E_size + *new_edge_size);
        uint32_t *new_V_sizeptr = thrust::max_element(thrust::device, new_V_size, new_V_size + *new_edge_size);
        numVertices = *new_V_sizeptr;
        numEdges = *new_E_sizeptr;

        Flag4[0]=1;
        CreateFlag4Array<<<numBlock,numthreads>>>(expanded_u, Flag4, numEdges);
        CreateNewVertexList<<<numBlock,numthreads>>>(VertexList, Flag4, numEdges, expanded_u);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: CreateNewVertexList%s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();
        d_hierarchy_levels.push_back(SuperVertexId);
        hierarchy_level_sizes.push_back(numVertices);
        cudaMallocManaged(&SuperVertexId, numVertices * sizeof(uint32_t));
    }
    std::string outFile="test";
    writeComponents(d_hierarchy_levels, image.rows*image.cols, 3, hierarchy_level_sizes, outFile, image.rows, image.cols);

    //Free memory
    cudaFree(flag);
    cudaFree(FlagList);
    cudaFree(VertexList);
    cudaFree(MinSegmentedList);
    cudaFree(tempArray);
    cudaFree(EdgeList);
    cudaFree(BitEdgeList);
    cudaFree(NWE);
    cudaFree(Successor);
    cudaFree(newSuccessor);
    cudaFree(OnlyEdge);
    cudaFree(OnlyWeight);
    cudaFree(L);
    cudaFree(Representative);
    cudaFree(VertexIds);
    cudaFree(new_E_size);
    cudaFree(new_V_size);
    cudaFree(MinMaxScanArray);
    cudaFree(compactLocations);
    cudaFree(expanded_u);
    cudaFree(C);
    cudaFree(Flag2);
    cudaFree(SuperVertexId);
    cudaFree(uid);
    cudaFree(flag4);
    cudaFree(UV);
    cudaFree(UVW);
    cudaFree(W);
    cudaFree(flag3);
    cudaFree(Flag4);
    for (int l = 0; l < d_hierarchy_levels.size(); l++) {
        cudaFree(d_hierarchy_levels[l]);
    }
    d_hierarchy_levels.clear();
    hierarchy_level_sizes.clear();
}

void printUsage() {
    puts("Usage: ./felz -i [input image path] -o [output image path]");
    puts("Options:");
    puts("\t-i: Path to input file (default: data/beach.png)");
    puts("\t-o: Path to output file (default: segmented.png)");
    puts("Benchmarking options");
    puts("\t-w: Number of iterations to perform during warmup");
    puts("\t-b: Number of iterations to perform during benchmarking");
    puts("\t-t: Timing mode: complete / parts (default complete)");
    exit(1);
}

const Options handleParams(int argc, char **argv) {
    Options options = Options();
        for(;;)
    {
        switch(getopt(argc, argv, "hi:o:w:b:t:"))
        {
        case 'i': {
            options.inFile = std::string(optarg);
            continue;
        }
        case 'o': {
            options.outFile = std::string(optarg);
            continue;
        }
        case 'w': {
            options.warmupIterations = atoi(optarg);
            continue;
        }
        case 'b': {
            options.benchmarkIterations = atoi(optarg);
            continue;
        }
        case 'p': {
            continue;
        }
        case '?':
        case 'h':
        default : {
            printUsage();
            break;
        }

        case -1:  {
            break;
        }
        }
        break;
    }/*
    if (options.inFile == "empty" || options.outFile == "empty") {
        puts("Provide an input and output image!");
        printUsage();
    }*/

    return options;
}


int main(int argc, char **argv)
{
    Mat image;
    const Options options = handleParams(argc, argv);

    image = imread("data/bear.jpg", IMREAD_COLOR);
    printf("Size of image obtained is: Rows: %d, Columns: %d, Pixels: %d\n", image.rows, image.cols, image.rows * image.cols);
    segment(image);

    return 0;
}
