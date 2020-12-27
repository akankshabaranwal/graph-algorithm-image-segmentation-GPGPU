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
enum timing_mode {NO_TIME, TIME_COMPLETE, TIME_PARTS};
enum timing_mode TIMING_MODE;
std::vector<int> timings;
bool NO_WRITE = false;

void ImagetoGraphParallelStream(Mat &image, uint32_t *d_vertex,uint32_t *d_edge, uint64_t *d_weight)
{
    std::chrono::high_resolution_clock::time_point start, end;

    GpuMat dev_image, d_blurred;; 	 // Released automatically in destructor
    cv::Ptr<cv::cuda::Filter> filter;

    if (TIMING_MODE == TIME_PARTS) { // Start gaussian filter timer
        start = std::chrono::high_resolution_clock::now();
    }

    // Apply gaussian filter
    dev_image.upload(image);
    filter = cv::cuda::createGaussianFilter(CV_8UC3, CV_8UC3, cv::Size(5, 5), 1.0);
    filter->apply(dev_image, d_blurred);

    if (TIMING_MODE == TIME_PARTS) { // End gaussian filter timer
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        int time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        timings.push_back(time);
    }

    if (TIMING_MODE == TIME_PARTS) { // Start graph creation timer
        start = std::chrono::high_resolution_clock::now();

    }

    // Create graphs. Kernels executed in different streams for concurrency
    dim3 encode_threads;
    dim3 encode_blocks;
    SetImageGridThreadLen(image.rows, image.cols, image.rows*image.cols, &encode_threads, &encode_blocks);

    int num_of_blocks, num_of_threads_per_block;

    SetGridThreadLen(image.cols, &num_of_blocks, &num_of_threads_per_block);
    dim3 grid_row(num_of_blocks, 1, 1);
    dim3 threads_row(num_of_threads_per_block, 1, 1);

    SetGridThreadLen(image.rows, &num_of_blocks, &num_of_threads_per_block);
    dim3 grid_col(num_of_blocks, 1, 1);
    dim3 threads_col(num_of_threads_per_block, 1, 1);

    dim3 grid_corner(1, 1, 1);
    dim3 threads_corner(4, 1, 1);

    size_t pitch = d_blurred.step;

    // Create inner graph
    createInnerGraphKernel<<< encode_blocks, encode_threads, 0>>>((unsigned char*) d_blurred.cudaPtr(), d_vertex, d_edge, d_weight, image.rows, image.cols, pitch);

    // Create outer graph
    createFirstRowGraphKernel<<< grid_row, threads_row, 1>>>((unsigned char*) d_blurred.cudaPtr(), d_vertex, d_edge, d_weight, image.rows, image.cols, pitch);
    createLastRowGraphKernel<<< grid_row, threads_row, 2>>>((unsigned char*) d_blurred.cudaPtr(), d_vertex, d_edge, d_weight, image.rows, image.cols, pitch);

    createFirstColumnGraphKernel<<< grid_col, threads_col, 3>>>((unsigned char*) d_blurred.cudaPtr(), d_vertex, d_edge, d_weight, image.rows, image.cols, pitch);
    createLastColumnGraphKernel<<< grid_col, threads_col, 4>>>((unsigned char*) d_blurred.cudaPtr(), d_vertex, d_edge, d_weight, image.rows, image.cols, pitch);

    // Create corners
    createCornerGraphKernel<<< grid_corner, threads_corner, 5>>>((unsigned char*) d_blurred.cudaPtr(), d_vertex, d_edge, d_weight, image.rows, image.cols, pitch);

    cudaDeviceSynchronize(); // Needed to synchronise streams!

    if (TIMING_MODE == TIME_PARTS) {
        end = std::chrono::high_resolution_clock::now();
        int time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        timings.push_back(time);
    }

}




void writeComponents(std::vector<uint32_t *>& d_hierarchy_levels, int no_of_vertices_orig, int Channel_size, std::vector<int>& hierarchy_level_sizes, std::string outFile, int no_of_rows, int no_of_cols) {
    // Extract filepath without extension
    size_t lastindex = outFile.find_last_of(".");
    std::string rawOutName = outFile.substr(0, lastindex);
    std::chrono::high_resolution_clock::time_point start, end;
    if (TIMING_MODE == TIME_PARTS || TIMING_MODE == TIME_COMPLETE) { // Start write timer
        start = std::chrono::high_resolution_clock::now();
    }

    // Generate random colors for segments
    char *component_colours = (char *) malloc(no_of_vertices_orig * Channel_size * sizeof(char));

    // Generate uniform [0, 1] float
    curandGenerator_t gen;
    char* d_component_colours;
    float *d_component_colours_float;
    cudaMalloc( (void**) &d_component_colours_float, no_of_vertices_orig * Channel_size * sizeof(float));
    cudaMalloc( (void**) &d_component_colours, no_of_vertices_orig * Channel_size * sizeof(char));

    // Generate random floats
    curandCreateGenerator(&gen , CURAND_RNG_PSEUDO_MTGP32); // Create a Mersenne Twister pseudorandom number generator
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); // Set seed
    curandGenerateUniform(gen, d_component_colours_float, no_of_vertices_orig * Channel_size); // Generate n floats on device

    // Convert floats to RGB char
    int num_of_blocks, num_of_threads_per_block;

    SetGridThreadLen(no_of_vertices_orig * Channel_size, &num_of_blocks, &num_of_threads_per_block);
    dim3 grid_rgb(num_of_blocks, 1, 1);
    dim3 threads_rgb(num_of_threads_per_block, 1, 1);

    RandFloatToRandRGB<<< grid_rgb, threads_rgb, 0>>>(d_component_colours, d_component_colours_float, no_of_vertices_orig * Channel_size);
    cudaFree(d_component_colours_float);

    // Create hierarchy
    unsigned int* d_prev_level_component;
    cudaMalloc((void**) &d_prev_level_component, sizeof(unsigned int)*no_of_vertices_orig);

    dim3 threads_pixels;
    dim3 grid_pixels;
    SetImageGridThreadLen(no_of_rows, no_of_cols, no_of_vertices_orig, &threads_pixels, &grid_pixels);

    InitPrevLevelComponents<<<grid_pixels, threads_pixels, 0>>>(d_prev_level_component, no_of_rows, no_of_cols);

    char* d_output_image;
    cudaMalloc( (void**) &d_output_image, no_of_rows*no_of_cols*Channel_size*sizeof(char));
    char *output = (char*) malloc(no_of_rows*no_of_cols*Channel_size*sizeof(char));

    for (int l = 0; l < d_hierarchy_levels.size(); l++) {
        int level_size = hierarchy_level_sizes[l];
        uint32_t* d_level = d_hierarchy_levels[l];

        CreateLevelOutput<<< grid_pixels, threads_pixels, 0>>>(d_output_image, d_component_colours, d_level, d_prev_level_component, no_of_rows, no_of_cols);
        cudaMemcpy(output, d_output_image, no_of_rows*no_of_cols*Channel_size*sizeof(char), cudaMemcpyDeviceToHost);

        cv::Mat output_img = cv::Mat(no_of_rows, no_of_cols, CV_8UC3, output);
        std::string outfilename = rawOutName + std::string("_")  + std::to_string(l) + std::string(".png");
        std::string outmessage = std::string("Writing ") + outfilename.c_str() + std::string("\n");
        if (!NO_WRITE) {
            cv::Mat output_img = cv::Mat(no_of_rows, no_of_cols, CV_8UC3, output);
            std::string outfilename = rawOutName + std::string("_")  + std::to_string(l) + std::string(".png");
            std::string outmessage = std::string("Writing ") + outfilename.c_str() + std::string("\n");

            fprintf(stderr, "%s", outmessage.c_str());
            imwrite(outfilename, output_img);
        }

    }


    if (TIMING_MODE == TIME_PARTS || TIMING_MODE == TIME_COMPLETE) { // End write timer
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        int time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (TIMING_MODE == TIME_PARTS) {
            timings.push_back(time);
        } else {
            timings[0] += time;
        }
    }

    // Free memory
    cudaFree(d_component_colours);
    cudaFree(d_prev_level_component);
    cudaFree(d_output_image);
    free(output);
}


void segment(Mat image, std::string outFile, bool output)
{
    int numVertices = image.rows * image.cols;
    uint numEdges = (image.rows) * (image.cols) * 4;
    std::chrono::high_resolution_clock::time_point start, end;

    if (TIMING_MODE == TIME_COMPLETE) { // Start whole execution timer
        start = std::chrono::high_resolution_clock::now();
    }

    //Convert image to graph
    uint32_t *VertexList, *OnlyEdge, *FlagList, *NWE, *Successor, *L, *Representative, *VertexIds;
    uint64_t *OnlyWeight, *tempArray, *BitEdgeList, *MinSegmentedList;
    uint32_t *MinMaxScanArray;
    uint32_t *new_E_size, *new_V_size;
    uint32_t *compactLocations, *expanded_u;
    uint32_t *C;
    edge *EdgeList;
    uint *flag;
    uint *Flag2;
    uint32_t *SuperVertexId;
    uint32_t *uid;
    uint *flag4;
    uint64_t *UVW;

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
    cudaMallocManaged(&uid, numEdges * sizeof(uint32_t));
    cudaMallocManaged(&flag4,numEdges * sizeof(uint));
    cudaMallocManaged(&UVW,numEdges*sizeof(uint64_t));
    cudaMallocManaged(&flag3,numEdges*sizeof(uint));
    cudaMallocManaged(&Flag4,numEdges*sizeof(uint));

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

    if (TIMING_MODE == TIME_PARTS) { // Start segmentation timer
        start = std::chrono::high_resolution_clock::now();
    }


    while(numVertices>1)
    {
        //printf("\nIn a new iteration with numVertices=%d, numEdges=%d\n", numVertices, numEdges);
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

        MarkSegments<<<numBlock, numthreads>>>(flag, VertexList, numVertices);

        //3. Segmented min scan
        SegmentedReduction(*context, VertexList, OnlyWeight, tempArray, numEdges, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: Segment Reduction%s\n", cudaGetErrorString(err));
            exit(-1);
        }

        cudaDeviceSynchronize();
        // Create NWE array
        CreateNWEArray<<<numBlock, numthreads>>>(NWE, tempArray, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: CreateNWEArray %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        cudaDeviceSynchronize();

        //4. Find Successor array of each vertex
        FindSuccessorArray<<<numBlock, numthreads>>>(Successor, BitEdgeList, NWE, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: FindSuccessorArray %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        //RemoveCycles
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
        }
        cudaDeviceSynchronize();

        thrust::sort_by_key( thrust::device,Representative, Representative + numVertices, VertexIds);
        //cudaDeviceSynchronize();

        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error ThrustSort Representative Array: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        cudaDeviceSynchronize();
        CreateFlag2Array<<<numBlock, numthreads>>>(Representative, Flag2, numVertices);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error CreateFlag2Array: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
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
        flag[0]=0;
        thrust::inclusive_scan(flag, flag + numEdges, uid, thrust::plus<int>());

        cudaDeviceSynchronize();

        //11. Removing self edges
        RemoveSelfEdges<<<numBlock,numthreads>>>(OnlyEdge, numEdges, uid, SuperVertexId);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error RemoveSelfEdges: %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        //E 12.
        CreateUVWArray<<<numBlock,numthreads>>>(BitEdgeList, OnlyEdge, numEdges, uid, SuperVertexId, UVW);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error CreateUVWArray: %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        //12.2 Sort the UVW Array
        thrust::sort(thrust::device, UVW, UVW + numEdges);
        cudaDeviceSynchronize();
        flag3[0]=1;
        CreateFlag3Array<<<numBlock,numthreads>>>(UVW, numEdges, flag3, MinMaxScanArray);

        uint32_t *new_edge_size = thrust::max_element(thrust::device, MinMaxScanArray, MinMaxScanArray + numEdges);
        cudaDeviceSynchronize();
        *new_edge_size = *new_edge_size+1;
        thrust::inclusive_scan(flag3, flag3 + *new_edge_size, compactLocations, thrust::plus<int>());

        ResetCompactLocationsArray<<<numBlock,numthreads>>>(compactLocations, *new_edge_size);
        CreateNewEdgeList<<<numBlock,numthreads>>>( BitEdgeList, compactLocations, OnlyEdge, OnlyWeight, UVW, flag3, *new_edge_size, new_E_size, new_V_size, expanded_u);
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error CreateNewEdgeList: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        uint32_t *new_E_sizeptr = thrust::max_element(thrust::device, new_E_size, new_E_size + *new_edge_size);
        uint32_t *new_V_sizeptr = thrust::max_element(thrust::device, new_V_size, new_V_size + *new_edge_size);
        cudaDeviceSynchronize();
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
        err = cudaGetLastError();        // Get error code
        if ( err != cudaSuccess )
        {
            printf("CUDA Error CudaMallocerror for SuperVertexIds: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
    if (TIMING_MODE == TIME_PARTS) { // End segmentation timer
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        int time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        timings.push_back(time);
    }

    if (TIMING_MODE == TIME_COMPLETE) { // End whole execution timer
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        int time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        timings.push_back(time);
    }

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
    cudaFree(UVW);
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
    TIMING_MODE=TIME_COMPLETE;
        for(;;)
    {
        switch(getopt(argc, argv, "pnhi:o:w:b:"))
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
            TIMING_MODE = TIME_PARTS;
            continue;
        }
        case 'n': {
            NO_WRITE = true;
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
    }
    if (options.inFile == "empty" || options.outFile == "empty") {
        puts("Provide an input and output image!");
        printUsage();
    }

    return options;
}
void printCSVHeader() {
    if (TIMING_MODE == TIME_COMPLETE) {
        printf("total\n"); // Excluding output: gaussian + graph creation + segmentation
    } else {
        printf("gaussian, graph, segmentation, output\n");
    }
}

void printCSVLine() {
    if (timings.size() > 0) {
        printf("%d", timings[0]);
        for (int i = 1; i < timings.size(); i++) {
            printf(",%d", timings[i]);
        }
        printf("\n");
        timings.clear();
    }
}

int main(int argc, char **argv)
{
    Mat image;
    const Options options = handleParams(argc, argv);

    image = imread(options.inFile, IMREAD_COLOR);
    //cv::resize(image, image, cv::Size(), 0.05, 0.05);
    //Scale down image.
    printf("Size of image obtained is: Rows: %d, Columns: %d, Pixels: %d\n", image.rows, image.cols, image.rows * image.cols);

    // Warm up
    for (int i = 0; i < options.warmupIterations; i++) {
        segment(image, options.outFile, false);
    }
    printCSVHeader();
    // Benchmark
    timings.clear();
    for (int i = 0; i < options.benchmarkIterations; i++) {
        segment(image, options.outFile, i == options.benchmarkIterations-1);
        printCSVLine();
    }

    return 0;
}
