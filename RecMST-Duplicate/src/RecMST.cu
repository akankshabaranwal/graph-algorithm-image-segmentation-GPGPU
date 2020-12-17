/***********************************************************************************
  Implementing Minimum Spanning Tree on CUDA using primitive operations for the 
  algorithm given in "Fast Minimum Spanning Tree Computation", by Pawan Harish, 
  P.J. Narayanan, Vibhav Vineet, and Suryakant Patidar.

  Chapter 7 of Nvidia GPU Computing Gems, Jade Edition, 2011.
  
  Copyright (c) 2011 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND, EXPRESSED, IMPLIED OR 
  OTHERWISE.

  Created by: Pawan Harish.
  Split Implementation by: Suryakant Patidar and Parikshit Sakurikar.
 ************************************************************************************/

/***********************************************************************************
  General bit size info
  ---------------------
  Vertex ID 26 bit -> 67.108.864
  - 8K image: 7680 × 4320 = 33.177.600 pixels -> supports 2 8K images

  Weight 12 bit -> Max weight = 4096
  - Max L2 distance RGB: 442 -> can use 3 more bits for extra precision (*8) (SCALE)
  - Could reduce weight precision to support higher resolution images

  1. Segmented min scan: 10 bit weight, 22 bit ID
  -> Changed to long long; 12 bit weight, 26 bit ID
  8. List L: 32 bit vertex ID left, 32 bit vertex ID right
  12. UVW: u.id 24 bit, v.id 24 bit, weight 16 bit
  -> Change to u.id 26 bit, v.id 26 bit, weight 12 bit
************************************************************************************/

////////////////////////////////////////////////
// Variables
////////////////////////////////////////////////

// Standard C stuff
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// C++ stuff
#include <iostream>
#include <vector>

// Command line options
#include <getopt.h>
#include "Options.h"

// Timings
#include <chrono>
#include <sys/time.h>

// Kernels
#include "Kernels.cu"

// Thrust stuff
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>

// Opencv stuff
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/cudafilters.hpp>
using namespace cv;
using namespace cv::cuda;

// Curand stuff
#include <cuda.h>
#include <curand.h>


////////////////////////////////////////////////
// Variables
////////////////////////////////////////////////
unsigned int no_of_rows;									// Number of rows in image
unsigned int no_of_cols;									// Number of columns in image

unsigned int no_of_vertices;								//Actual input graph sizes
unsigned int no_of_vertices_orig;							//Original number of vertices graph (constant)

unsigned int no_of_edges;									//Current graph sizes
unsigned int no_of_edges_orig;								//Original number of edges graph (constant)

unsigned int *d_edge;										// Starts as h_edge
unsigned int *d_vertex;										// starts as h_vertex
unsigned int *d_weight;										// starts as h_weight

unsigned long long int *d_segmented_min_scan_input;			//X, Input to the Segmented Min Scan, appended array of weights and edge IDs
unsigned long long int *d_segmented_min_scan_output;		//Output of the Segmented Min Scan, minimum weight outgoing edge as (weight|to_vertex_id elements) can be found at end of each segment
unsigned int *d_edge_flag;									//Flag for the segmented min scan
unsigned int *d_edge_flag_thrust;							//NEW! Flag for the segmented min scan in thrust Needs to be 000111222 instead of 100100100
unsigned int *d_vertex_flag;								//F2, Flag for the scan input for supervertex ID generation
unsigned int *d_pick_array;									//PickArray for each edge. index min weight outgoing edge of u in sorted array if not removed. Else -1 if removed (representative doesn't add edges)
unsigned int *d_successor;									//S, Successor Array
unsigned int *d_successor_copy;								//Helper array for pointer doubling
bool *d_succchange;											//Variable to check if can stop pointer doubling

unsigned int *d_new_supervertexIDs;							//mapping from each original vertex ID to its new supervertex ID so we can lookup supervertex IDs directly
unsigned int *d_old_uIDs;									//expanded old u ids, stored per edge, needed to remove self edges (orig ID of source vertex u for each edge(weight|dest_vertex_id_v))
unsigned long long int *d_appended_uvw;						//Appended u,v,w array for duplicate edge removal

unsigned int *d_size;										//Stores amount of edges
unsigned int *d_edge_mapping_copy;
unsigned int *d_edge_list_size;
unsigned int *d_vertex_list_size;

unsigned long long int *d_vertex_split;						//L, Input to the split function

// Hierarchy output
int cur_hierarchy_size; 									// Size current hierarchy

enum timing_mode {NO_TIME, TIME_COMPLETE, TIME_PARTS};
enum timing_mode TIMING_MODE;
std::vector<int> timings;

////////////////////////////////////////////////
// Debugging helper functions
////////////////////////////////////////////////
void printIntArr(int* d_data, int n_elements) {
	int* h_data = (int *)malloc(sizeof(int)*n_elements);
	cudaMemcpy(h_data, d_data, sizeof(int) * n_elements, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n_elements; i++) {
		printf("%d ",h_data[i]);
	}
	printf("\n");
	free(h_data);
}

void printXArr(int* d_data, int n_elements) {
	int* h_data = (int *)malloc(sizeof(int)*n_elements);
	cudaMemcpy(h_data, d_data, sizeof(int) * n_elements, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n_elements; i++) {
		int mask = pow(2.0,MOVEBITS)-1;
		int vertex = h_data[i]&mask;
		int weight = h_data[i]>>MOVEBITS;
		printf("%d|%d ",weight, vertex);
	}
	printf("\n");
	free(h_data);
}

void printUVWArr(unsigned long long int *d_data, int n_elements) {
	unsigned long long int* h_data = (unsigned long long int *)malloc(sizeof(unsigned long long int)*n_elements);
	cudaMemcpy(h_data, d_data, sizeof(unsigned long long int) * n_elements, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n_elements; i++) {
		unsigned long long int UVW = h_data[i];
		unsigned long long int mask = pow(2.0,64-(NO_OF_BITS_MOVED_FOR_VERTEX_IDS+NO_OF_BITS_MOVED_FOR_VERTEX_IDS))-1;
		unsigned long long int w  = (int) UVW&mask;
		unsigned long long int test = UVW>>(64-(NO_OF_BITS_MOVED_FOR_VERTEX_IDS+NO_OF_BITS_MOVED_FOR_VERTEX_IDS));
		unsigned long long int mask2 = pow(2.0,NO_OF_BITS_MOVED_FOR_VERTEX_IDS)-1;
		unsigned long long int v = test&mask2;
		unsigned long long int u = test>>NO_OF_BITS_MOVED_FOR_VERTEX_IDS;
		printf("%llu|%llu|%llu ",u, v, w);
	}
	printf("\n");
	free(h_data);
}

void printUIntArr(unsigned int* d_data, int n_elements) {
	unsigned int* h_data = (unsigned int *)malloc(sizeof(unsigned int)*n_elements);
	cudaMemcpy(h_data, d_data, sizeof(unsigned int) * n_elements, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n_elements; i++) {
		printf("%u ",h_data[i]);
	}
	printf("\n");
	free(h_data);
}

void printULongArr(long* d_data, int n_elements) {
	unsigned long* h_data = (unsigned long *)malloc(sizeof(unsigned long)*n_elements);
	cudaMemcpy(h_data, d_data, sizeof(unsigned long) * n_elements, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n_elements; i++) {
		printf("%lu ",h_data[i]);
	}
	printf("\n");
	free(h_data);
}

void printLongArr(long* d_data, int n_elements) {
	long* h_data = (long *)malloc(sizeof(long)*n_elements);
	cudaMemcpy(h_data, d_data, sizeof(long) * n_elements, cudaMemcpyDeviceToHost);
	for (int i = 0; i < n_elements; i++) {
		printf("%ld ",h_data[i]);
	}
	printf("\n");
	free(h_data);
}

void printInt(int *d_val) {
	int h_val;
	cudaMemcpy( &h_val, d_val, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d", h_val);
}

void printUInt(unsigned int *d_val) {
	unsigned int h_val;
	cudaMemcpy( &h_val, d_val, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	printf("%u", h_val);
}

////////////////////////////////////////////////
// Helper functions to set the grid sizes
////////////////////////////////////////////////
void SetGridThreadLen(int number, int *num_of_blocks, int *num_of_threads_per_block)
{
	*num_of_blocks = 1;
	*num_of_threads_per_block = number;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(number>MAX_THREADS_PER_BLOCK)
	{
		*num_of_blocks = (int)ceil(number/(double)MAX_THREADS_PER_BLOCK); 
		*num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}
}

void SetImageGridThreadLen(int no_of_rows, int no_of_cols, int no_of_vertices, dim3* encode_threads, dim3* encode_blocks)
{    
    if (no_of_vertices < 1024) {
        encode_threads->x = no_of_rows;
        encode_threads->y = no_of_cols;
        encode_blocks->x = 1;
        encode_blocks->y = 1;
    } else {
        encode_threads->x = 32;
        encode_threads->y = 32;
        encode_blocks->x = no_of_rows / 32 + 1;
        encode_blocks->y = no_of_cols / 32 + 1;
    }
}

////////////////////////////////////////////////
// Allocate and Free segmentation Arrays
////////////////////////////////////////////////
void Init()
{

	//Allocate graph device memory
	cudaMalloc( (void**) &d_edge, sizeof(unsigned int)*no_of_edges_orig);
	cudaMalloc( (void**) &d_vertex, sizeof(unsigned int)*no_of_vertices_orig);
	cudaMalloc( (void**) &d_weight, sizeof(unsigned int)*no_of_edges_orig);

	//Allocate memory for other arrays
	cudaMalloc( (void**) &d_segmented_min_scan_input, sizeof(unsigned long long int)*no_of_edges_orig);
	cudaMalloc( (void**) &d_segmented_min_scan_output, sizeof(unsigned long long int)*no_of_edges_orig);
	cudaMalloc( (void**) &d_edge_flag, sizeof(unsigned int)*no_of_edges_orig);
	cudaMalloc( (void**) &d_edge_flag_thrust, sizeof(unsigned int)*no_of_edges_orig);
	cudaMalloc( (void**) &d_pick_array, sizeof(unsigned int)*no_of_edges_orig);
	cudaMalloc( (void**) &d_successor,sizeof(unsigned int)*no_of_vertices_orig);
	cudaMalloc( (void**) &d_successor_copy,sizeof(unsigned int)*no_of_vertices_orig);
	
	//Clear Output MST array
	cudaMalloc( (void**) &d_succchange, sizeof(bool));
	cudaMalloc( (void**) &d_vertex_split, sizeof(unsigned long long int)*no_of_vertices_orig);
	cudaMalloc( (void**) &d_vertex_flag, sizeof(unsigned int)*no_of_vertices_orig);
	cudaMalloc( (void**) &d_new_supervertexIDs, sizeof(unsigned int)*no_of_vertices_orig);
	cudaMalloc( (void**) &d_old_uIDs, sizeof(unsigned int)*no_of_edges_orig);
	cudaMalloc( (void**) &d_appended_uvw, sizeof(unsigned long long int)*no_of_edges_orig);
	cudaMalloc( (void**) &d_size, sizeof(unsigned int));
	cudaMalloc( (void**) &d_edge_mapping_copy, sizeof(unsigned int)*no_of_edges_orig); 

	cudaMalloc( (void**) &d_edge_list_size, sizeof(unsigned int));
	cudaMalloc( (void**) &d_vertex_list_size, sizeof(unsigned int));
	
}

void FreeMem()
{
	cudaFree(d_edge);
	cudaFree(d_vertex);
	cudaFree(d_weight);
	cudaFree(d_segmented_min_scan_input);
	cudaFree(d_segmented_min_scan_output);
	cudaFree(d_edge_flag);
	cudaFree(d_edge_flag_thrust);
	cudaFree(d_pick_array);
	cudaFree(d_successor);
	cudaFree(d_successor_copy);
	cudaFree(d_succchange);
	cudaFree(d_vertex_split);
	cudaFree(d_vertex_flag);
	cudaFree(d_new_supervertexIDs);
	cudaFree(d_old_uIDs);
	cudaFree(d_size);
	cudaFree(d_edge_mapping_copy);
	cudaFree(d_edge_list_size);
	cudaFree(d_vertex_list_size);
	cudaFree(d_appended_uvw);
}

////////////////////////////////////////////////
// Create graph in compressed adjacency list
////////////////////////////////////////////////
void createGraph(Mat image) {
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

	// Allocate GPU segmentation memory
	Init();


	// Create graphs. Kernels executed in different streams for concurrency
	dim3 encode_threads;
	dim3 encode_blocks;
	SetImageGridThreadLen(no_of_rows, no_of_cols, no_of_vertices_orig, &encode_threads, &encode_blocks);

    int num_of_blocks, num_of_threads_per_block;

	SetGridThreadLen(no_of_cols, &num_of_blocks, &num_of_threads_per_block);
	dim3 grid_row(num_of_blocks, 1, 1);
	dim3 threads_row(num_of_threads_per_block, 1, 1);

	SetGridThreadLen(no_of_rows, &num_of_blocks, &num_of_threads_per_block);
	dim3 grid_col(num_of_blocks, 1, 1);
	dim3 threads_col(num_of_threads_per_block, 1, 1);

    dim3 grid_corner(1, 1, 1);
	dim3 threads_corner(4, 1, 1);

    size_t pitch = d_blurred.step;

    // Create inner graph
    createInnerGraphKernel<<< encode_blocks, encode_threads, 0>>>((unsigned char*) d_blurred.cudaPtr(), d_vertex, d_edge, d_weight, no_of_rows, no_of_cols, pitch);

    // Create outer graph
   	createFirstRowGraphKernel<<< grid_row, threads_row, 1>>>((unsigned char*) d_blurred.cudaPtr(), d_vertex, d_edge, d_weight, no_of_rows, no_of_cols, pitch);
   	createLastRowGraphKernel<<< grid_row, threads_row, 2>>>((unsigned char*) d_blurred.cudaPtr(), d_vertex, d_edge, d_weight, no_of_rows, no_of_cols, pitch);

   	createFirstColumnGraphKernel<<< grid_col, threads_col, 3>>>((unsigned char*) d_blurred.cudaPtr(), d_vertex, d_edge, d_weight, no_of_rows, no_of_cols, pitch);
   	createLastColumnGraphKernel<<< grid_col, threads_col, 4>>>((unsigned char*) d_blurred.cudaPtr(), d_vertex, d_edge, d_weight, no_of_rows, no_of_cols, pitch);

    // Create corners
	createCornerGraphKernel<<< grid_corner, threads_corner, 5>>>((unsigned char*) d_blurred.cudaPtr(), d_vertex, d_edge, d_weight, no_of_rows, no_of_cols, pitch);
	
	cudaDeviceSynchronize(); // Needed to synchronise streams!

	if (TIMING_MODE == TIME_PARTS) {
		end = std::chrono::high_resolution_clock::now();
		int time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		timings.push_back(time);
	}


	fprintf(stderr, "Image read successfully into graph with %d vertices and %d edges\n", no_of_vertices, no_of_edges);
}


////////////////////////////////////////////////
// Perform Our Recursive MST Algorithm
////////////////////////////////////////////////
void HPGMST()
{
	//Make both CUDA grids needed for execution, no_of_vertices and no_of_edges length sizes
	int num_of_blocks, num_of_threads_per_block;

	//Grid and block sizes so each edge has one thread (fit as much threads as possible in one block)
	SetGridThreadLen(no_of_edges, &num_of_blocks, &num_of_threads_per_block);
	dim3 grid_edgelen(num_of_blocks, 1, 1);
	dim3 threads_edgelen(num_of_threads_per_block, 1, 1);

	// Grid and block sizes so each vertex has one thread (fit as much threads as possible in one block)
	SetGridThreadLen(no_of_vertices, &num_of_blocks, &num_of_threads_per_block);
	dim3 grid_vertexlen(num_of_blocks, 1, 1);
	dim3 threads_vertexlen(num_of_threads_per_block, 1, 1);

	/*
	 * A. Find minimum weighted edge
	 */

	// 1. Append weight w and outgoing vertex v per edge into a single array, X.
    // 12 bit for weight, 26 bits for ID.
	//Append in Parallel on the Device itself, call the append kernel
	AppendKernel_1<<< grid_edgelen, threads_edgelen, 0>>>(d_segmented_min_scan_input, d_weight, d_edge, no_of_edges);

	// d_edge_flag = F
	//Create the Flag needed for segmented min scan operation, similar operation will also be used at other places
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_flag, no_of_edges );


	// 2. Divide the edge-list, E, into segments with 1 indicating the start of each segment and 0 otherwise, store this in flag array F.
	// Mark the segments for the segmented min scan
	MakeFlag_3<<< grid_vertexlen, threads_vertexlen, 0>>>( d_edge_flag, d_vertex, no_of_vertices);


	// 3. Perform segmented min scan on X with F indicating segments to find minimum outgoing edge-index per vertex. Min can be found at end of each segment after scan // DONE: change to thrust
	// Prepare key vector for thrust
	thrust::inclusive_scan(thrust::device, d_edge_flag, d_edge_flag + no_of_edges, d_edge_flag_thrust);

	// Min inclusive segmented scan on ints from start to end.
	thrust::equal_to<unsigned int> binaryPred;
	thrust::minimum<unsigned long long int> binaryOp;
	thrust::inclusive_scan_by_key(thrust::device, d_edge_flag_thrust, d_edge_flag_thrust + no_of_edges, d_segmented_min_scan_input, d_segmented_min_scan_output, binaryPred, binaryOp);


	/*
	 * B. Finding and removing cycles
	 */

	// 4. Find the successor of each vertex and add to successor array, S.
	MakeSucessorArray<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_vertex, d_segmented_min_scan_output, no_of_vertices, no_of_edges);


	// 5. Remove cycle making edges using S, and identify representatives vertices.
	RemoveCycles<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor,no_of_vertices);


	/*
	 * Can possibly be moved in future once remove pick array stuff
	 */
	//Scan the flag to get u at every edge, use the u to index d_vertex to get the last entry in each segment
	//U at every edge will also be useful later in the algorithm.

	// Set F[0] = 0. F is the same as previous F but first element is 0 instead of 1
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_flag, no_of_edges );
	MakeFlagForUIds<<< grid_vertexlen, threads_vertexlen, 0>>>(d_edge_flag, d_vertex,no_of_vertices); 

	// 10.2 Create vector indicating source vertex u for each edge // DONE: change to thrust
	thrust::inclusive_scan(thrust::device, d_edge_flag, d_edge_flag + no_of_edges, d_old_uIDs);


	/*
	 * C. Merging vertices and assigning IDs to supervertices
	 */


	// 7. Propagate Representative Vertex IDs to all vertices iteratively using pointer Doubling until no change occures in Successor Array
	bool succchange;
	do
	{
		succchange=false; //if no thread changes this value, the loop stops
		cudaMemcpy( d_succchange, &succchange, sizeof(bool), cudaMemcpyHostToDevice);
		//Reusing Vertex Flag
		SuccToCopy<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_successor_copy, no_of_vertices); // for conflicts
		PropagateRepresentativeID<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_successor_copy, d_succchange,no_of_vertices);
		CopyToSucc<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_successor_copy, no_of_vertices); // for conflicts

		cudaMemcpy( &succchange, d_succchange, sizeof(bool), cudaMemcpyDeviceToHost);
	}
	while(succchange);


	// 8. Append successor array’s entries with its index to form a list, L. Representative left, vertex id right, 64 bit.
	//    Append Vertex Ids with SuperVertexIDs
	AppendVertexIDsForSplit<<< grid_vertexlen, threads_vertexlen, 0>>>(d_vertex_split, d_successor,no_of_vertices);


	//9. Split L, create flag over split output and scan the flag to find new ids per vertex, store new ids in C.
    // 9.1 Split L using representative as key. In parallel using a split of O(V) with log(V) bit key size.
    //     split based on supervertex IDs using 64 bit version of split
	thrust::sort(thrust::device, d_vertex_split, d_vertex_split + no_of_vertices);


	// 9.2 Create flag for assigning new vertex IDs based on difference in supervertex IDs
	//     first element not flagged so that can use simple sum for scan
	ClearArray<<< grid_vertexlen, threads_vertexlen, 0>>>( d_vertex_flag, no_of_vertices);
	MakeFlagForScan<<< grid_vertexlen, threads_vertexlen, 0>>>(d_vertex_flag, d_vertex_split, no_of_vertices);

	// 9.3 Scan flag to assign new IDs to supervertices, Using a scan on O(V) elements // DONE: change to thrust
	thrust::inclusive_scan(thrust::device, d_vertex_flag, d_vertex_flag + no_of_vertices, d_new_supervertexIDs);


	/*
	 * D. Removing self edges
	 */

	// 10.1 Create mapping from each original vertex ID to its new supervertex ID so we can lookup supervertex IDs directly
	MakeSuperVertexIDPerVertex<<< grid_vertexlen, threads_vertexlen, 0>>>(d_new_supervertexIDs, d_vertex_split, d_vertex_flag, no_of_vertices);
	CopySuperVertexIDPerVertex<<< grid_vertexlen, threads_vertexlen, 0>>>(d_new_supervertexIDs, d_vertex_flag, no_of_vertices); // for concurrent access problems
	
	//Remove Self Edges from the edge-list
	// 11. Remove edge from edge-list if u, v have same supervertex id (remove self edges)
	CopyEdgeArray<<< grid_edgelen, threads_edgelen, 0>>>(d_edge,d_edge_mapping_copy, no_of_edges); // for conflicts
	RemoveSelfEdges<<< grid_edgelen, threads_edgelen, 0>>>(d_edge, d_old_uIDs, d_new_supervertexIDs, d_edge_mapping_copy, no_of_edges);
	CopyEdgeArrayBack<<< grid_edgelen, threads_edgelen, 0>>>(d_edge,d_edge_mapping_copy, no_of_edges); // for conflicts

	/*
	 * D. Removing duplicate edges. This is not mandatory, however, reduces the edge-list size significantly. You may choose to use it once in the initial 
	 *    iterations of the algorithm, later edge-list size is small anyways so not much is gained by doing this in later iterations
	 */


	// 12. Remove the largest duplicate edges using split over new u,v and w.
	// 12.1 Append supervertex ids of u and v along with weight w into single 64 bit array (u 24 bit, v 24 bit, w 16 bit)
	AppendForDuplicateEdgeRemoval<<< grid_edgelen, threads_edgelen, 0>>>(d_appended_uvw, d_edge, d_old_uIDs, d_weight,d_new_supervertexIDs, no_of_edges);

	//12.2 Split the array using {u,v) as the key. Pick First distinct (u,v) entry as the edge, nullify others
	//     You may also replace the split with sort, but we could not find a 64-bit sort.
	thrust::sort(thrust::device, d_appended_uvw, d_appended_uvw + no_of_edges);
	
	//Pick the first distinct (u,v) combination, mark these edges and compact
	// 12.3 Create flag indicating smallest edges, 0 for larger duplicates
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_flag, no_of_edges ); // d_edge_flag = F3
	unsigned int dsize=no_of_edges; //just make sure
	cudaMemcpy( d_size, &dsize, sizeof(unsigned int), cudaMemcpyHostToDevice);
	MarkEdgesUV<<< grid_edgelen, threads_edgelen, 0>>>(d_edge_flag, d_appended_uvw, d_size, no_of_edges);


	// 13. Compact and create new edge and weight list
	// 13.1 Scan the flag array to know where to write the value in new edge and weight lists // DONE: change to thrust
	thrust::inclusive_scan(thrust::device, d_edge_flag, d_edge_flag + no_of_edges, d_old_uIDs);

	// Make sure new locations start from 0 instead of 1.
	thrust::transform(thrust::device,
				  d_old_uIDs,
                  d_old_uIDs + no_of_edges,
                  thrust::make_constant_iterator(1),
                  d_old_uIDs,
                  thrust::minus<unsigned int>());


	// Do some cleanup / clearing
	ClearEdgeStuff<<< grid_edgelen, threads_edgelen, 0>>>((unsigned int*)d_edge, (unsigned int*)d_weight, d_edge_mapping_copy, (unsigned int*)d_pick_array, no_of_edges);
	unsigned int negative=0;
	cudaMemcpy( d_edge_list_size, &negative, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy( d_vertex_list_size, &negative, sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	//Compact the edge and weight lists
	unsigned int validsize=0;
	cudaMemcpy( &validsize, d_size, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	//Make a new grid for valid entries in the d_edge_flag array
	SetGridThreadLen(validsize, &num_of_blocks, &num_of_threads_per_block);
	dim3 grid_validsizelen(num_of_blocks, 1, 1);
	dim3 threads_validsizelen(num_of_threads_per_block, 1, 1);

	// 13.2 Compact and create new edge and weight list
	//      Reusing d_pick_array for storing the u ids
	CompactEdgeList<<< grid_validsizelen, threads_validsizelen, 0>>>(d_edge, d_weight, d_old_uIDs, d_edge_flag, d_appended_uvw, d_pick_array, d_size, d_edge_list_size, d_vertex_list_size);

	// 14. Build the vertex list from the newly formed edge list
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_flag, no_of_edges);
	ClearArray<<< grid_vertexlen, threads_vertexlen, 0>>>((unsigned int*)d_vertex, no_of_vertices);

	//14.1 Create flag based on difference in u on the new edge list (based on diffference of u ids)
	MakeFlagForVertexList<<< grid_edgelen, threads_edgelen, 0>>>(d_pick_array, d_edge_flag, no_of_edges); // d_edge_flag = F4

	// 14.2 Build the vertex list from the newly formed edge list
	MakeVertexList<<< grid_edgelen, threads_edgelen, 0>>>(d_vertex, d_pick_array, d_edge_flag, no_of_edges);
	
	cur_hierarchy_size = no_of_vertices;
	cudaMemcpy( &no_of_edges, d_edge_list_size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy( &no_of_vertices, d_vertex_list_size, sizeof(unsigned int), cudaMemcpyDeviceToHost);

}


void writeComponents(std::vector<unsigned int*>& d_hierarchy_levels, std::vector<int>& hierarchy_level_sizes, std::string outFile) {
	// Extract filepath without extension
	size_t lastindex = outFile.find_last_of("."); 
	std::string rawOutName = outFile.substr(0, lastindex);

	// Generate random colors for segments
	char *component_colours = (char *) malloc(no_of_vertices_orig * CHANNEL_SIZE * sizeof(char));


	// Generate uniform [0, 1] float
	curandGenerator_t gen;
	char* d_component_colours;
	float *d_component_colours_float;
	cudaMalloc( (void**) &d_component_colours_float, no_of_vertices_orig * CHANNEL_SIZE * sizeof(float));
	cudaMalloc( (void**) &d_component_colours, no_of_vertices_orig * CHANNEL_SIZE * sizeof(char));

	// Generate random floats
	curandCreateGenerator(&gen , CURAND_RNG_PSEUDO_MTGP32); // Create a Mersenne Twister pseudorandom number generator
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); // Set seed
	curandGenerateUniform(gen, d_component_colours_float, no_of_vertices_orig * CHANNEL_SIZE); // Generate n floats on device

	// Convert floats to RGB char
	int num_of_blocks, num_of_threads_per_block;

	SetGridThreadLen(no_of_vertices_orig * CHANNEL_SIZE, &num_of_blocks, &num_of_threads_per_block);
	dim3 grid_rgb(num_of_blocks, 1, 1);
	dim3 threads_rgb(num_of_threads_per_block, 1, 1);

	RandFloatToRandRGB<<< grid_rgb, threads_rgb, 0>>>(d_component_colours, d_component_colours_float, no_of_vertices_orig * CHANNEL_SIZE);
	cudaFree(d_component_colours_float);


	// Create hierarchy
	unsigned int* d_prev_level_component;
	cudaMalloc((void**) &d_prev_level_component, sizeof(unsigned int)*no_of_vertices_orig);

	dim3 threads_pixels;
    dim3 grid_pixels;
	SetImageGridThreadLen(no_of_rows, no_of_cols, no_of_vertices_orig, &threads_pixels, &grid_pixels);

    InitPrevLevelComponents<<<grid_pixels, threads_pixels, 0>>>(d_prev_level_component, no_of_rows, no_of_cols);

    char* d_output_image;
	cudaMalloc( (void**) &d_output_image, no_of_rows*no_of_cols*CHANNEL_SIZE*sizeof(char));
    char *output = (char*) malloc(no_of_rows*no_of_cols*CHANNEL_SIZE*sizeof(char));

    for (int l = 0; l < d_hierarchy_levels.size(); l++) {
		int level_size = hierarchy_level_sizes[l];
		unsigned int* d_level = d_hierarchy_levels[l];

		CreateLevelOutput<<< grid_pixels, threads_pixels, 0>>>(d_output_image, d_component_colours, d_level, d_prev_level_component, no_of_rows, no_of_cols);
	    cudaMemcpy(output, d_output_image, no_of_rows*no_of_cols*CHANNEL_SIZE*sizeof(char), cudaMemcpyDeviceToHost);

		cv::Mat output_img = cv::Mat(no_of_rows, no_of_cols, CV_8UC3, output);
		std::string outfilename = rawOutName + std::string("_")  + std::to_string(l) + std::string(".png");
		std::string outmessage = std::string("Writing ") + outfilename.c_str() + std::string("\n");

		fprintf(stderr, "%s", outmessage.c_str());
		imwrite(outfilename, output_img);
	}


	// Free memory
	cudaFree(d_component_colours);
	cudaFree(d_prev_level_component);
	cudaFree(d_output_image);
	free(output);
}

void setGraphParams(unsigned int rows, unsigned int cols) {
	no_of_rows = rows;
    no_of_cols = cols;
	no_of_vertices = no_of_rows * no_of_cols;
	no_of_vertices_orig = no_of_vertices;
	no_of_edges = 8 + 6 * (no_of_cols - 2) + 6 * (no_of_rows - 2) + 4 * (no_of_cols - 2) * (no_of_rows - 2);
	no_of_edges_orig = no_of_edges;
}

void clearHierarchy(std::vector<unsigned int*>& d_hierarchy_levels, std::vector<int>& hierarchy_level_sizes) {
	for (int l = 0; l < d_hierarchy_levels.size(); l++) {
			cudaFree(d_hierarchy_levels[l]);
		}
        d_hierarchy_levels.clear();
        hierarchy_level_sizes.clear();
}

void segment(Mat image, std::string outFile, bool output) {
	std::chrono::high_resolution_clock::time_point start, end;

	if (TIMING_MODE == TIME_COMPLETE) { // Start whole execution timer
		start = std::chrono::high_resolution_clock::now();
	}


	// Reset num vertices in edges in case of multiple iterations
	no_of_edges = no_of_edges_orig;
	no_of_vertices = no_of_vertices_orig;

	std::vector<unsigned int*> d_hierarchy_levels;	// Vector containing pointers to all hierarchy levels (don't dereference on CPU, device pointers)
	std::vector<int> hierarchy_level_sizes;			// Size of each hierarchy level

	// Graph creation
	createGraph(image);


	if (TIMING_MODE == TIME_PARTS) { // Start segmentation timer
		start = std::chrono::high_resolution_clock::now();
	}
	
	// Segmentation
	do
	{
	    HPGMST();

	    d_hierarchy_levels.push_back(d_new_supervertexIDs);
	    hierarchy_level_sizes.push_back(cur_hierarchy_size);
	    cudaMalloc( (void**) &d_new_supervertexIDs, sizeof(unsigned int)*cur_hierarchy_size);

	    fprintf(stderr, "Vertices: %d\n", no_of_vertices);
	}
	while(no_of_vertices>1);

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

	// Free GPU segmentation memory
	FreeMem();

	if (output) {
		// Write segmentation hierarchy
		writeComponents(d_hierarchy_levels, hierarchy_level_sizes, outFile);
	}

	clearHierarchy(d_hierarchy_levels, hierarchy_level_sizes);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
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

void printCSVHeader() {
	if (TIMING_MODE == TIME_COMPLETE) {
		 printf("total\n"); // Excluding output: gaussian + graph creation + segmentation
	} else {
		printf("gaussian, graph creation, segmentation\n");
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

const Options handleParams(int argc, char **argv) {
    Options options = Options();
    TIMING_MODE = TIME_COMPLETE;
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
                TIMING_MODE = TIME_PARTS;
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

int main(int argc, char **argv)
{
    const Options options = handleParams(argc, argv);

    // Read image
    Mat image = imread(options.inFile, IMREAD_COLOR);
    fprintf(stderr, "Size of image obtained is: Rows: %d, Columns: %d, Pixels: %d\n", image.rows, image.cols, image.rows * image.cols);
   	setGraphParams(image.rows, image.cols);

   	printCSVHeader();

	// Warm up
    for (int i = 0; i < options.warmupIterations; i++) {
    	segment(image, options.outFile, false);
    }

    // Benchmark
    timings.clear();
    for (int i = 0; i < options.benchmarkIterations; i++) {
        segment(image, options.outFile, i == options.benchmarkIterations-1);
        printCSVLine();
    }

    return 0;
}

