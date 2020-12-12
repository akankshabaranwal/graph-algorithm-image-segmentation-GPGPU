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
Vertex ID bit size:
- 8K image: 33.177.600 pixels, 26 bits = 67.108.864 pixels 
-> Gives 12 bits for weight -> max weight = 4096
	- Max weight L2 distance: 442, can use 3 more bits (* 2 * 2 * 2)
	  - multiply double * 8, then scale to int (be wary for further modifications)



1. Segmented min scan: 10 bit weight, 22 bit ID
   -> Change to long long 12 bit weight, 26 bit ID + add last 20 bit of ID to weight for tiebreaking cycles
8. List L: 32 bit vertex ID left, 32 bit vertex ID right
   -> keep same
12. UVW: u.id 24 bit, v.id 24 bit, weight 16 bit
   -> Change to u.id 26 bit, v.id 26 bit, weight 12 bit
************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <vector>

// includes, project
// #include <cutil.h> // Removed, should just have been for CUDA_SAFE_CALL and CUDA_CUT_CALL which has been deprecated

// includes, kernels
#include "Kernels.cu"
// #include <cudpp.h> 

// Thrust stuff
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>

// Opencv stuff
#include <opencv2/highgui.hpp>
#include <opencv2/cudafilters.hpp>
using namespace cv;
using namespace cv::cuda;

#define CHANNEL_SIZE 3

////////////////////////////////////////////////
// Variables
////////////////////////////////////////////////
unsigned int no_of_rows;										// Number of rows in image
unsigned int no_of_cols;										// Number of columns in image

unsigned int no_of_vertices;									//Actual input graph sizes
unsigned int no_of_vertices_orig;							//Original number of vertices graph (constant)

unsigned int no_of_edges;									//Current graph sizes
unsigned int no_of_edges_orig;								//Original number of edges graph (constant)

//Graph held in these variables at the host end
unsigned int *h_edge;										//Original E (end vertex index each edge)
unsigned int *h_vertex;										//Original V (start index edges for each vertex)
unsigned int *h_weight;										//Original W (weight each edge)

//Graph held in these variables at the device end
unsigned int *d_edge;										// Starts as h_edge
unsigned int *d_vertex;										// starts as h_vertex
unsigned int *d_weight;										// starts as h_weight

unsigned int *d_segmented_min_scan_input;					//X, Input to the Segmented Min Scan, appended array of weights and edge IDs
unsigned int *d_segmented_min_scan_output;					//Output of the Segmented Min Scan, minimum weight outgoing edge as (weight|to_vertex_id elements) can be found at end of each segment
unsigned int *d_edge_flag;							//Flag for the segmented min scan
unsigned int *d_edge_flag_thrust;					//NEW! Flag for the segmented min scan in thrust Needs to be 000111222 instead of 100100100
unsigned int *d_vertex_flag;						//F2, Flag for the scan input for supervertex ID generation
unsigned int *d_pick_array;									//PickArray for each edge. index min weight outgoing edge of u in sorted array if not removed. Else -1 if removed (representative doesn't add edges)
unsigned int *d_successor;									//S, Successor Array
unsigned int *d_successor_copy;								//Helper array for pointer doubling
bool *d_succchange;									//Variable to check if can stop pointer doubling

unsigned int *d_new_supervertexIDs;					//mapping from each original vertex ID to its new supervertex ID so we can lookup supervertex IDs directly
unsigned int *d_old_uIDs;							//expanded old u ids, stored per edge, needed to remove self edges (orig ID of source vertex u for each edge(weight|dest_vertex_id_v))
unsigned long long int *d_appended_uvw;				//Appended u,v,w array for duplicate edge removal

unsigned int *d_size;								//Stores amount of edges
unsigned int *d_edge_mapping_copy;
unsigned int	*d_edge_list_size;
unsigned int	*d_vertex_list_size;

unsigned long long int *d_vertex_split;				//L, Input to the split function

// Hierarchy output
int cur_hierarchy_size; // Size current hierarchy
std::vector<unsigned int*> hierarchy_levels;// Vector containing pointers to all hierarchy levels
std::vector<int> hierarchy_level_sizes;// Size of each hierarchy level

// Debug helper function
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
// Read the Graph in our format (Compressed adjacency list)
////////////////////////////////////////////////
unsigned int dissimilarity(Mat image, int row1, int col1, int row2, int col2) {
    Point3_<uchar>* u = image.ptr<Point3_<uchar> >(row1,col1);
    Point3_<uchar>* v = image.ptr<Point3_<uchar> >(row2,col2);
    double distance = sqrt(pow((u->x - v->x), 2) + pow((u->y - v->y), 2) + pow((u->z - v->z), 2));
    return (unsigned int) round(distance); // TODO: maybe map to larger interval for better accuracy
}


int ImagetoGraphSerial(Mat image) {
    
	int cur_edge_idx, cur_vertex_idx, left_node, right_node, bottom_node, top_node;
    cur_edge_idx = 0;

    for(int i=0; i<image.rows; i++) {
        for(int j=0; j<image.cols; j++) {
            left_node = i * image.cols + j - 1;
            right_node = i * image.cols + j + 1;
            bottom_node = (i+1) * image.cols + j;
            top_node = (i - 1) * image.cols + j;

            //Add the index for VertexList
            cur_vertex_idx = i * image.cols + j;
            h_vertex[cur_vertex_idx] = cur_edge_idx;

            if (j > 0) {
            	h_edge[cur_edge_idx] = left_node;
            	h_weight[cur_edge_idx] = dissimilarity(image, i, j, i, j - 1);
                cur_edge_idx++;
            }

            if (j < image.cols - 1) {
                h_edge[cur_edge_idx] = right_node;
                h_weight[cur_edge_idx] = dissimilarity(image, i, j, i, j + 1);
                cur_edge_idx++;
            }

            if (i < image.rows - 1) {
                h_edge[cur_edge_idx] = bottom_node;
                h_weight[cur_edge_idx] = dissimilarity(image, i, j, i+1, j);
                cur_edge_idx++;
            }

            if (i > 0) {
                h_edge[cur_edge_idx] = top_node;
                h_weight[cur_edge_idx] = dissimilarity(image, i, j, i-1, j);
                cur_edge_idx++;
            }
        }
    }

    return cur_edge_idx;
}


void ReadGraph(char *filename) {

	Mat image, output;				// Released automatically
    GpuMat dev_image, dev_output; 	// Released automatically

    // Read image
    image = imread(filename, IMREAD_COLOR);
    printf("Size of image obtained is: Rows: %d, Columns: %d, Pixels: %d\n", image.rows, image.cols, image.rows * image.cols);
    no_of_rows = image.rows;
    no_of_cols = image.cols;

    // Apply gaussian filter
    dev_image.upload(image);
    Ptr<Filter> filter = createGaussianFilter(CV_8UC3, CV_8UC3, Size(5, 5), 1.0);
    filter->apply(dev_image, dev_output);
    dev_output.download(output);

    // TODO: use dev_output for gaussian filter

    // Get graph parameters
	no_of_vertices = image.rows * image.cols;
	no_of_vertices_orig = no_of_vertices;
	h_vertex = (unsigned int*)malloc(sizeof(unsigned int)*no_of_vertices);

	// Initial approximation number of edges
	no_of_edges = (image.rows)*(image.cols)*4;
	h_edge = (unsigned int*) malloc (sizeof(unsigned int)*no_of_edges);
	h_weight = (unsigned int*) malloc (sizeof(unsigned int)*no_of_edges);

	no_of_edges = ImagetoGraphSerial(image);
	no_of_edges_orig = no_of_edges;

	// Scale down to real size
	h_edge = (unsigned int*) realloc(h_edge, no_of_edges * sizeof(unsigned int));
	h_weight = (unsigned int*) realloc(h_weight, no_of_edges * sizeof(unsigned int));

	printf("Image read successfully into graph with %d vertices and %d edges\n", no_of_vertices, no_of_edges);
}


////////////////////////////////////////////////
// Allocate and Initialize Arrays
////////////////////////////////////////////////
void Init()
{

	//Copy the Graph to Device
	cudaMalloc( (void**) &d_edge, sizeof(unsigned int)*no_of_edges);
	cudaMalloc( (void**) &d_vertex, sizeof(unsigned int)*no_of_vertices);
	cudaMalloc( (void**) &d_weight, sizeof(unsigned int)*no_of_edges);
	cudaMemcpy( d_edge, h_edge, sizeof(unsigned int)*no_of_edges, cudaMemcpyHostToDevice);
	cudaMemcpy( d_vertex, h_vertex, sizeof(unsigned int)*no_of_vertices, cudaMemcpyHostToDevice);
	cudaMemcpy( d_weight, h_weight, sizeof(unsigned int)*no_of_edges, cudaMemcpyHostToDevice);
	printf("Graph Copied to Device\n");

	//Allocate memory for other arrays
	cudaMalloc( (void**) &d_segmented_min_scan_input, sizeof(unsigned int)*no_of_edges);
	cudaMalloc( (void**) &d_segmented_min_scan_output, sizeof(unsigned int)*no_of_edges);
	cudaMalloc( (void**) &d_edge_flag, sizeof(unsigned int)*no_of_edges);
	cudaMalloc( (void**) &d_edge_flag_thrust, sizeof(unsigned int)*no_of_edges);
	cudaMalloc( (void**) &d_pick_array, sizeof(unsigned int)*no_of_edges);
	cudaMalloc( (void**) &d_successor,sizeof(unsigned int)*no_of_vertices);
	cudaMalloc( (void**) &d_successor_copy,sizeof(unsigned int)*no_of_vertices);
	
	//Clear Output MST array
	cudaMalloc( (void**) &d_succchange, sizeof(bool));
	cudaMalloc( (void**) &d_vertex_split, sizeof(unsigned long long int)*no_of_vertices);
	cudaMalloc( (void**) &d_vertex_flag, sizeof(unsigned int)*no_of_vertices);
	cudaMalloc( (void**) &d_new_supervertexIDs, sizeof(unsigned int)*no_of_vertices);
	cudaMalloc( (void**) &d_old_uIDs, sizeof(unsigned int)*no_of_edges);
	cudaMalloc( (void**) &d_appended_uvw, sizeof(unsigned long long int)*no_of_edges);
	cudaMalloc( (void**) &d_size, sizeof(unsigned int));
	cudaMalloc( (void**) &d_edge_mapping_copy, sizeof(unsigned int)*no_of_edges); 

	cudaMalloc( (void**) &d_edge_list_size, sizeof(unsigned int));
	cudaMalloc( (void**) &d_vertex_list_size, sizeof(unsigned int));
	
}


////////////////////////////////////////////////
// Helper function to set the grid sizes
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
    // Normally 8-10 bit for weight, 20-22 bits for ID. Because of 32 bit limitation CUDPP scan primitive, TODO: probably not relevant anymore
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

	//printf("X:\n");
	//printUIntArr(d_edge_flag, no_of_edges);
	//printXArr(d_segmented_min_scan_input, no_of_edges);

	// Min inclusive segmented scan on ints from start to end.
	thrust::equal_to<unsigned int> binaryPred;
	thrust::minimum<unsigned int> binaryOp;
	thrust::inclusive_scan_by_key(thrust::device, d_edge_flag_thrust, d_edge_flag_thrust + no_of_edges, d_segmented_min_scan_input, d_segmented_min_scan_output, binaryPred, binaryOp);

	//printXArr(d_segmented_min_scan_output, no_of_edges);
	//printf("\n");


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

	//printf("Expanded U:\n");
	//printUIntArr(d_old_uIDs, no_of_edges);


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


	// 8. Append successor array’s entries with its index to form a list, L. Representative left, vertex id right, 64 bit. TODO look into needed sizes
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
	//printf("New supervertex ids:\n");
	thrust::inclusive_scan(thrust::device, d_vertex_flag, d_vertex_flag + no_of_vertices, d_new_supervertexIDs);
	//printUIntArr(d_new_supervertexIDs, no_of_vertices);


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
	thrust::sort(thrust::device, d_appended_uvw, d_appended_uvw + no_of_edges); // TODO: check
	
	//Pick the first distinct (u,v) combination, mark these edges and compact
	// 12.3 Create flag indicating smallest edges, 0 for larger duplicates
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_flag, no_of_edges ); // d_edge_flag = F3
	unsigned int dsize=no_of_edges; //just make sure
	cudaMemcpy( d_size, &dsize, sizeof(unsigned int), cudaMemcpyHostToDevice);
	MarkEdgesUV<<< grid_edgelen, threads_edgelen, 0>>>(d_edge_flag, d_appended_uvw, d_size, no_of_edges);

	//printf("UVW:");
	//printUVWArr(d_appended_uvw, no_of_edges);
	//printUIntArr(d_edge_flag, no_of_edges);

	//printf("New edge size: ");
	//printUInt(d_size);

	// 13. Compact and create new edge and weight list
	// 13.1 Scan the flag array to know where to write the value in new edge and weight lists // DONE: change to thrust
	thrust::inclusive_scan(thrust::device, d_edge_flag, d_edge_flag + no_of_edges, d_old_uIDs);

	// NEW! Maybe not needed. Make sure new locations start from 0 instead of 1. TODO: can be done more efficient in case works
	thrust::transform(thrust::device,
				  d_old_uIDs,
                  d_old_uIDs + no_of_edges,
                  thrust::make_constant_iterator(1),
                  d_old_uIDs,
                  thrust::minus<unsigned int>());

	//printf("Write positions:");

	//******************************************************************************************
	//Do all clearing in a single kernel, no need to call multiple times, OK for testing only TODO
	//******************************************************************************************
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>((unsigned int*)d_edge, no_of_edges );
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>((unsigned int*)d_weight, no_of_edges );
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_mapping_copy, no_of_edges);
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( (unsigned int*)d_pick_array, no_of_edges); //Reusing the Pick Array
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



////////////////////////////////////////////////
//Free All memory from Host and Device
////////////////////////////////////////////////
void FreeMem()
{
	cudaFree(d_edge);
	cudaFree(d_vertex);
	cudaFree(d_weight);
	cudaFree(d_segmented_min_scan_input);
	cudaFree(d_segmented_min_scan_output);
	cudaFree(d_edge_flag);
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

	free(h_edge);
	free(h_vertex);
	free(h_weight);
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
void get_component_colours(char colours[], uint num_colours) {
    srand(123456789);
    for (int i = 0; i < num_colours * CHANNEL_SIZE; i++) {
        colours[i] = rand() % 256;
    }
}


int main( int argc, char** argv) {
	if(argc<2) {
		printf("Specify an Input Image\n");
		exit(1);
	}


	ReadGraph(argv[1]);
	Init();
	//printf("\n\n");

	/*unsigned int	timer;
	cutCreateTimer( &timer);	
	cutStartTimer( timer);*/
	//Perform Our MST algorhtm
	printf("start\n");
	do
	{
	    HPGMST();

	    // Add hierarchy level
	    unsigned int* cur_hierarchy = (unsigned int*)malloc(sizeof(unsigned int)*cur_hierarchy_size);
	    cudaMemcpy(cur_hierarchy, d_new_supervertexIDs, sizeof(unsigned int)*cur_hierarchy_size, cudaMemcpyDeviceToHost);

	    hierarchy_levels.push_back(cur_hierarchy);
	    hierarchy_level_sizes.push_back(cur_hierarchy_size);
	    
	    printf("%d\n", no_of_vertices);
	}
	while(no_of_vertices>1);


	// Write back hierarchy output
	// Generate random colors for segments
	char *component_colours = (char *) malloc(no_of_vertices_orig * CHANNEL_SIZE * sizeof(char));
	get_component_colours(component_colours, no_of_vertices_orig);

	char *output = (char*) malloc(no_of_rows*no_of_rows*CHANNEL_SIZE*sizeof(char));

	unsigned int* prev_level_component = (unsigned int*)malloc(sizeof(unsigned int)*no_of_vertices_orig);
	for (int i = 0; i < no_of_rows; i++) {
		for (int j = 0; j < no_of_cols; j++) {
			prev_level_component[i * no_of_cols + j] = i * no_of_cols + j;
		}
	}

	for (int l = 0; l < hierarchy_levels.size(); l++) {
		int level_size = hierarchy_level_sizes[l];
		unsigned int* level = hierarchy_levels[l];
		for (int i = 0; i < no_of_rows; i++) {
			for (int j = 0; j < no_of_cols; j++) {
				unsigned int prev_component = prev_level_component[i * no_of_cols + j];
				unsigned int new_component = level[prev_component];

				int img_pos = CHANNEL_SIZE * (i * no_of_cols + j);
				int colour_pos = CHANNEL_SIZE * new_component;
				output[img_pos] = component_colours[colour_pos];
				output[img_pos + 1] = component_colours[colour_pos+1];
				output[img_pos + 2] = component_colours[colour_pos+2];

                prev_level_component[i * no_of_cols + j] = new_component;
			}
		}
		cv::Mat output_img = cv::Mat(no_of_rows, no_of_cols, CV_8UC3, output);
		printf("Writing segmented_%d.png\n", l);
		imwrite("segmented_" + std::to_string(l) + ".png", output_img);
	}


	/*cutStopTimer( timer);
	printf("\n=================== Time taken To perform MST :: %3.3f ms===================\n",cutGetTimerValue(timer));*/
	//printf("\n\nOutputs:\n");

	//Copy the Final MST array to the CPU memory, a 1 at the index means that edge was selected in the MST, 0 otherwise.
	//It should be noted that each edge has an opposite edge also, out of whcih only one is selected in this output.
	//So total number of 1s in this array must be equal to no_of_vertices_orig-1.
	/*int k=0;
	int weight=0;
	//printf("\n\nSelected Edges in MST...\n\n");
	for(int i=0;i<no_of_edges_orig;i++)
		if(h_output_MST_test[i]==1)
			{
				printf("%d %d\n",h_edge[i],h_weight[i]);
				k++;
				weight+=h_weight[i];
			}
		//else {
		//	printf("not %d %d\n",h_edge[i],h_weight[i]);
		//}
	printf("\nNumber of edges in MST, must be=(no_of_vertices-1)): %d,(%d)\nTotal MST weight: %d\n",k, no_of_vertices_orig,weight);*/
	printf("Done\n");
	FreeMem();
	return 0;
}

