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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include "Kernels.cu"
// #include <cudpp.h> // DONE: remove
#include "splitFuncs.h"
splitSort sp;

// Thrust stuff
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>


////////////////////////////////////////////////
// Variables
////////////////////////////////////////////////
int no_of_vertices,no_of_vertices_orig;				//Actual input graph sizes
int no_of_edges, no_of_edges_orig;					//Current graph sizes
int *h_edge, *h_vertex, *h_weight;					//Graph held in these variables at the host end, 3 arrays for compressed adjacency list format
int *d_edge, *d_vertex, *d_weight;					//Graph held in these variables at the device end
int *d_segmented_min_scan_input;					//Input to the Segmented Min Scan, appended array of weights and edge IDs (X in paper)
int *d_segmented_min_scan_output;					//Output of the Segmented Min Scan, minimum weight outgoing edge as (weight|to_vertex_id elements) for each verte
unsigned int *d_edge_flag;							//Flag for the segmented min scan
unsigned int *d_edge_flag_thrust;					//NEW! Flag for the segmented min scan in thrust Needs to be 000111222 instead of 100100100
unsigned int *d_vertex_flag;						//Flag for the scan input for supervertex ID generation
unsigned int *d_output_MST;							//Final output, marks 1 for selected edges in MST, 0 otherwise
int *d_pick_array;									//PickArray for each edge. For each edge from u, segmented scan location min edge going out of u if not removed. Else -1 if removed (representative doesn't add edges)
int *d_successor;									//Successor Array, S
int *d_successor_copy;
bool *d_succchange;									//Variable to check for execution while propagating representative vertex IDs
unsigned long long int *d_vertex_split;				//Input to the split function
unsigned long long int *d_vertex_split_scratchmem;	//Scratch memory to the split function
unsigned long long int *d_vertex_split_rank;		//Ranking arrary to the split function
unsigned long long int *d_vertex_rank_scratchmem;	//Scratch memory to the split function
unsigned int *d_new_supervertexIDs;					//new supervertex ids after scanning older IDs
unsigned int *d_old_uIDs;							//old ids, stored per edge, needed to remove self edges (orig ID of source vertex u for each edge(weight|dest_vertex_id_v))
unsigned long long int *d_appended_uvw;				//Appended u,v,w array for duplicate edge removal
unsigned long long int *d_edge_split_scratchmem;	//Scratch memory to the split function
unsigned long long int *d_edge_rank;				//Rank array for duplicate edge removal
unsigned long long int *d_edge_rank_scratchmem;		//Scratch memory to the split function
unsigned int *d_size;								//Stores amount of edges
unsigned int *d_edge_mapping;
unsigned int *d_edge_mapping_copy;
int	*d_edge_list_size;
int	*d_vertex_list_size;

unsigned int *h_output_MST_test;					//Final output on host, marks 1 for selected edges in MST, 0 otherwise
unsigned long long int *h_vertex_split_rank_test;	//Used to copy split rank to device, initially 1 2 3 4 5 ...
unsigned long long int *h_edge_rank_test;			//Used to copy edge rank to device, initially 1 2 3 4 5 ...

//CUDPP Scan and Segmented Scan Variables
// CUDPPHandle			segmentedScanPlan_min, scanPlan_add ;   // DONE: remove
// CUDPPConfiguration	config_segmented_min, config_scan_add ; // DONE: remove


////////////////////////////////////////////////
// Read the Graph in our format (Compressed adjacency list)
////////////////////////////////////////////////
void ReadGraph(char *filename)
{
	FILE *fp;
	fp = fopen(filename,"r");

	// Read vertices into array
	fscanf(fp,"%d",&no_of_vertices);
	h_vertex = (int*)malloc(sizeof(int)*no_of_vertices);
	no_of_vertices_orig = no_of_vertices ;
	int start, index ;
	for ( int i = 0 ; i < no_of_vertices ; i++ )
	{
		fscanf(fp,"%d %d",&start, &index) ; // Format: start edges, ignored
		h_vertex[i] = start ;
	}

	// Read "root" of graph
	int source = 0 ;
	fscanf(fp,"%d",&source);

	// Read edges of graph
	fscanf(fp,"%d",&no_of_edges);

	no_of_edges_orig = no_of_edges ;

	h_edge = (int*) malloc (sizeof(int)*no_of_edges);
	h_weight = (int*) malloc (sizeof(int)*no_of_edges);

	int edgeindex, edgeweight ;
	for( int i = 0 ; i < no_of_edges ; i++ )
	{
		fscanf(fp,"%d %d",&edgeindex, &edgeweight); // Format: to, weight
		h_edge[i] = edgeindex ;
		h_weight[i] =  edgeweight ;
	}
	fclose(fp);
	printf("File read successfully %d %d\n",no_of_vertices, no_of_edges);
}


////////////////////////////////////////////////
// Allocate and Initialize Arrays
////////////////////////////////////////////////
void Init()
{

	/*
	//Setting the CUDPP configurations for SCAN and SEGMENTED MIN SCAN // DONE: remove
	// Min inclusive segmented scan on ints from start to end.
	config_segmented_min.algorithm = CUDPP_SEGMENTED_SCAN;
	config_segmented_min.op = CUDPP_MIN;
	config_segmented_min.datatype = CUDPP_INT;
	config_segmented_min.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

	// Summation scan on ints from start to end. Each summation sums elements up to the current element i
	config_scan_add.algorithm = CUDPP_SCAN;
	config_scan_add.op = CUDPP_ADD;
	config_scan_add.datatype = CUDPP_INT;
	config_scan_add.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
	*/

	//Copy the Graph to Device
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_edge, sizeof(int)*no_of_edges));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_vertex, sizeof(int)*no_of_vertices));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_weight, sizeof(int)*no_of_edges));
	CUDA_SAFE_CALL( cudaMemcpy( d_edge, h_edge, sizeof(int)*no_of_edges, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy( d_vertex, h_vertex, sizeof(int)*no_of_vertices, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy( d_weight, h_weight, sizeof(int)*no_of_edges, cudaMemcpyHostToDevice));
	printf("Graph Copied to Device\n");

	//Allocate memory for other arrays
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_segmented_min_scan_input, sizeof(int)*no_of_edges));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_segmented_min_scan_output, sizeof(int)*no_of_edges));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_edge_flag, sizeof(unsigned int)*no_of_edges));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_edge_flag_thrust, sizeof(unsigned int)*no_of_edges));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_pick_array, sizeof(unsigned int)*no_of_edges));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_successor,sizeof(int)*no_of_vertices));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_successor_copy,sizeof(int)*no_of_vertices));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_output_MST, sizeof(unsigned int)*no_of_edges));
	
	//Clear Output MST array
	unsigned int *h_test=(unsigned int*)malloc(sizeof(unsigned int)*no_of_edges);
	for(int i=0;i<no_of_edges;i++)h_test[i]=0;
	CUDA_SAFE_CALL( cudaMemcpy( d_output_MST, h_test, sizeof(unsigned int)*no_of_edges, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_succchange, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_vertex_split, sizeof(unsigned long long int)*no_of_vertices));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_vertex_split_scratchmem, sizeof(unsigned long long int)*no_of_vertices));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_vertex_flag, sizeof(unsigned int)*no_of_vertices));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_new_supervertexIDs, sizeof(unsigned int)*no_of_vertices));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_old_uIDs, sizeof(unsigned int)*no_of_edges));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_appended_uvw, sizeof(unsigned long long int)*no_of_edges));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_edge_split_scratchmem, sizeof(unsigned long long int)*no_of_edges));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_size, sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_edge_mapping, sizeof(unsigned int)*no_of_edges)); 
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_edge_mapping_copy, sizeof(unsigned int)*no_of_edges)); 
	//Initiaize the d_edge_mapping array
	for(int i=0;i<no_of_edges;i++)h_test[i]=i;
	CUDA_SAFE_CALL( cudaMemcpy( d_edge_mapping, h_test, sizeof(unsigned int)*no_of_edges, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_edge_list_size, sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_vertex_list_size, sizeof(int)));
	

	h_output_MST_test = (unsigned int*)malloc(sizeof(unsigned int)*no_of_edges);

	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_vertex_split_rank, sizeof(unsigned long long int)*no_of_vertices));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_vertex_rank_scratchmem, sizeof(unsigned long long int)*no_of_vertices));
	h_vertex_split_rank_test=(unsigned long long int*)malloc(sizeof(unsigned long long int)*no_of_vertices);
	for(int i=0;i<no_of_vertices;i++)h_vertex_split_rank_test[i]=i;
	CUDA_SAFE_CALL( cudaMemcpy( d_vertex_split_rank, h_vertex_split_rank_test, sizeof(unsigned long long int)*no_of_vertices, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_edge_rank, sizeof(unsigned long long int)*no_of_edges));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_edge_rank_scratchmem, sizeof(unsigned long long int)*no_of_edges));
	//Initialize the edge rank list
	h_edge_rank_test=(unsigned long long int*)malloc(sizeof(unsigned long long int)*no_of_edges);
	for(int i=0;i<no_of_edges;i++)h_edge_rank_test[i]=i;
	CUDA_SAFE_CALL( cudaMemcpy( d_edge_rank, h_edge_rank_test, sizeof(unsigned long long int)*no_of_edges, cudaMemcpyHostToDevice));

	free(h_test);
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
	
	//Reinitialize the ranking arrays, must be orig but this also works
	CUDA_SAFE_CALL( cudaMemcpy( d_vertex_split_rank, h_vertex_split_rank_test, sizeof(unsigned long long int)*no_of_vertices, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy( d_edge_rank, h_edge_rank_test, sizeof(unsigned long long int)*no_of_edges, cudaMemcpyHostToDevice));
	

	//Make both grids needed for execution, no_of_vertices and no_of_edges length sizes
	int num_of_blocks, num_of_threads_per_block;

	// Grid and block sizes so each edge has one thread (fit as much threads as possible in one block)
	SetGridThreadLen(no_of_edges, &num_of_blocks, &num_of_threads_per_block);
	dim3 grid_edgelen(num_of_blocks, 1, 1);
	dim3 threads_edgelen(num_of_threads_per_block, 1, 1);

	// Grid and block sizes so each vertex has one thread (fit as much threads as possible in one block)
	SetGridThreadLen(no_of_vertices, &num_of_blocks, &num_of_threads_per_block);
	dim3 grid_vertexlen(num_of_blocks, 1, 1);
	dim3 threads_vertexlen(num_of_threads_per_block, 1, 1);


	//Append the Weight and Outgoing vertex into a single array, 8-10 bits for weight and 20-22 bits for vertex ID
	//Append in Parallel on the Device itself, call the append kernel
	AppendKernel_1<<< grid_edgelen, threads_edgelen, 0>>>(d_segmented_min_scan_input, d_weight, d_edge, no_of_edges);
	
	//Create the Flag needed for segmented min scan operation, similar operation will also be used at other places
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_flag, no_of_edges );

	//Mark the segments for the segmented min scan using scan
	MakeFlag_3<<< grid_vertexlen, threads_vertexlen, 0>>>( d_edge_flag, d_vertex, no_of_vertices);

	//Perfom the Segmented Min Scan on resulting array using d_edge_flag as segments // DONE: change to thrust
	// // Min inclusive segmented scan on ints from start to end.
	thrust::inclusive_scan(thrust::device, d_edge_flag, d_edge_flag + no_of_edges, d_edge_flag_thrust);

	thrust::equal_to<unsigned int> binaryPred;
	thrust::minimum<unsigned int> binaryOp;
	thrust::inclusive_scan_by_key(thrust::device, d_edge_flag_thrust, d_edge_flag_thrust + no_of_edges, d_segmented_min_scan_input, d_segmented_min_scan_output, binaryPred, binaryOp);

	/*
	cudppPlan(&segmentedScanPlan_min, config_segmented_min, no_of_edges, 1, 0 ); //Make the segmented min scan plan
	cudppSegmentedScan(segmentedScanPlan_min, d_segmented_min_scan_output, d_segmented_min_scan_input, (const unsigned int*)d_edge_flag, no_of_edges);
	cudppDestroyPlan(segmentedScanPlan_min);
	*/

	//Make the Sucessor Array
	MakeSucessorArray<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_vertex, d_segmented_min_scan_output, no_of_vertices, no_of_edges);

	//Check and remove if cycles exist
	RemoveCycles<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor,no_of_vertices);

	//Scan the flag to get u at every edge, use the u to index d_vertex to get the last entry in each segment
	//U at every edge will also be useful later in the algorithm.
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_flag, no_of_edges ); // Set F to 0 again 

	MakeFlagForUIds<<< grid_vertexlen, threads_vertexlen, 0>>>(d_edge_flag, d_vertex,no_of_vertices); // F is now same as previous F but first element is 0 instead of 1

	// For each edge calculate vertex ID u of source vertex // DONE: change to thrust
	thrust::inclusive_scan(thrust::device, d_edge_flag, d_edge_flag + no_of_edges, d_old_uIDs);

	/*
	cudppPlan(&scanPlan_add, config_scan_add, no_of_edges , 1, 0);// Create scanplan 
	cudppScan(scanPlan_add, d_old_uIDs, d_edge_flag, no_of_edges);
	cudppDestroyPlan(scanPlan_add);
	*/

	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>((unsigned int*)d_pick_array, no_of_edges);

	//Fill the pick array using the above and the d_successor array
 	MakePickArray<<< grid_edgelen, threads_edgelen, 0>>>(d_pick_array,d_successor,d_vertex,d_old_uIDs,no_of_vertices,no_of_edges);

	//Mark the Remaining Edges in the Output MST array. This not so elegant.
	//Because we do not know which edge index was selected by the segmented min scan,
	//we check each edge with the selected edges and write to output if same
	MarkOutputEdges<<< grid_edgelen, threads_edgelen, 0>>>(d_pick_array, d_segmented_min_scan_input, d_segmented_min_scan_output, d_output_MST,d_edge_mapping,no_of_edges);

	//Propagate Representative Vertex IDs to all vertices iteratively using pointer Doubling until no change occures in Successor Array
	bool succchange;
	do
	{
		succchange=false; //if no thread changes this value, the loop stops
		CUDA_SAFE_CALL( cudaMemcpy( d_succchange, &succchange, sizeof(bool), cudaMemcpyHostToDevice));
		//Reusing Vertex Flag
		SuccToCopy<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_successor_copy, no_of_vertices);
		PropagateRepresentativeID<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_successor_copy, d_succchange,no_of_vertices);
		CopyToSucc<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_successor_copy, no_of_vertices);
		CUDA_SAFE_CALL( cudaMemcpy( &succchange, d_succchange, sizeof(bool), cudaMemcpyDeviceToHost));
	}
	while(succchange);


	//Split the vertex ids based on supervertex ids
	//Using 64 bit version of split
	//Append Vertex Ids with SuperVertexIDs and split based on supervertex IDs
	AppendVertexIDsForSplit<<< grid_vertexlen, threads_vertexlen, 0>>>(d_vertex_split, d_successor,no_of_vertices);

	sp.split(d_vertex_split, d_vertex_split_rank, d_vertex_split_scratchmem, d_vertex_rank_scratchmem, no_of_vertices, NO_OF_BITS_TO_SPLIT_ON, 0);


	//Make the flag for assigning new vertex IDs based on difference in SuperVertex IDs
	ClearArray<<< grid_vertexlen, threads_vertexlen, 0>>>( d_vertex_flag, no_of_vertices);
	MakeFlagForScan<<< grid_vertexlen, threads_vertexlen, 0>>>(d_vertex_flag, d_vertex_split, no_of_vertices);

	//Scan the newly formed flag array to assign new ids to supervertices // DONE: change to thrust
	thrust::inclusive_scan(thrust::device, d_vertex_flag, d_vertex_flag + no_of_vertices, d_new_supervertexIDs);
	/*
	cudppPlan(&scanPlan_add, config_scan_add, no_of_vertices , 1, 0);
	cudppScan(scanPlan_add, d_new_supervertexIDs, d_vertex_flag, no_of_vertices);
	cudppDestroyPlan(scanPlan_add);
	*/

	//Make the new supervertexids per vertex
	MakeSuperVertexIDPerVertex<<< grid_vertexlen, threads_vertexlen, 0>>>(d_new_supervertexIDs, d_vertex_split, d_vertex_flag, no_of_vertices);

	CopySuperVertexIDPerVertex<<< grid_vertexlen, threads_vertexlen, 0>>>(d_new_supervertexIDs, d_vertex_flag, no_of_vertices);
	
	//Remove Self Edges from the edge-list
	CopyEdgeArray<<< grid_edgelen, threads_edgelen, 0>>>(d_edge,d_edge_mapping_copy, no_of_edges);
	RemoveSelfEdges<<< grid_edgelen, threads_edgelen, 0>>>(d_edge, d_old_uIDs, d_new_supervertexIDs, d_vertex_split_rank, d_edge_mapping_copy, no_of_edges);
	CopyEdgeArrayBack<<< grid_edgelen, threads_edgelen, 0>>>(d_edge,d_edge_mapping_copy, no_of_edges);

	//Now, Remove Duplicated Edges. This is not mandatory, however, reduces the edge-list size
	//significantly. You may choose to use it once in the initial iterations of the algorithm, later edge-list
	//size is small anyways so not much is gained by doing this in later iterations
	//Append u,v and weight per edge in a 64-bit value
	//Split the array using {u,v) as the key. Pick First distinct (u,v) entry as the edge, nullify others

	//You may also replace the split with sort, but we could not find a 64-bit sort.
	AppendForDuplicateEdgeRemoval<<< grid_edgelen, threads_edgelen, 0>>>(d_appended_uvw, d_edge, d_old_uIDs, d_weight,d_new_supervertexIDs, no_of_edges);

	sp.split(d_appended_uvw, d_edge_rank, d_edge_split_scratchmem, d_edge_rank_scratchmem, no_of_edges, NO_OF_BITS_TO_SPLIT_ON_UVW, 0);
	
	//Pick the first distinct (u,v) combination, mark these edges and compact
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_flag, no_of_edges );
	unsigned int dsize=no_of_edges; //just make sure
	CUDA_SAFE_CALL( cudaMemcpy( d_size, &dsize, sizeof(unsigned int), cudaMemcpyHostToDevice));
	MarkEdgesUV<<< grid_edgelen, threads_edgelen, 0>>>(d_edge_flag, d_appended_uvw, d_size, no_of_edges);

	//Scan the flag array to know where to write the value in new edge and weight lists // DONE: change to thrust
	thrust::inclusive_scan(thrust::device, d_edge_flag, d_edge_flag + no_of_edges, d_old_uIDs);
	/*
	cudppPlan(&scanPlan_add, config_scan_add, no_of_edges, 1, 0);
	cudppScan(scanPlan_add, d_old_uIDs, d_edge_flag, no_of_edges); //Just reusing the d_old_uIDs array for compating
	cudppDestroyPlan(scanPlan_add);
	*/

	//******************************************************************************************
	//Do all clearing in a single kernel, no need to call multiple times, OK for testing only
	//******************************************************************************************
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>((unsigned int*)d_edge, no_of_edges );
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>((unsigned int*)d_weight, no_of_edges );
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_mapping_copy, no_of_edges);
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( (unsigned int*)d_pick_array, no_of_edges); //Reusing the Pick Array
	int negative=0;
	CUDA_SAFE_CALL( cudaMemcpy( d_edge_list_size, &negative, sizeof( int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy( d_vertex_list_size, &negative, sizeof( int), cudaMemcpyHostToDevice));
	
	//Compact the edge and weight lists
	//Make a new grid for valid entries in the d_edge_flag array
	unsigned int validsize=0;
	CUDA_SAFE_CALL( cudaMemcpy( &validsize, d_size, sizeof(unsigned int), cudaMemcpyDeviceToHost));

	SetGridThreadLen(validsize, &num_of_blocks, &num_of_threads_per_block);
	dim3 grid_validsizelen(num_of_blocks, 1, 1);
	dim3 threads_validsizelen(num_of_threads_per_block, 1, 1);
	//Reusing d_pick_array for storing the u ids
	CompactEdgeList<<< grid_validsizelen, threads_validsizelen, 0>>>(d_edge, d_weight, d_edge_mapping, d_edge_mapping_copy, d_old_uIDs, d_edge_flag, d_appended_uvw, d_pick_array, d_edge_rank, d_size, d_edge_list_size, d_vertex_list_size);
	CopyEdgeMap<<< grid_edgelen, threads_edgelen, 0>>>(d_edge_mapping, d_edge_mapping_copy,no_of_edges);

	//Make the vertex list
	//Mark flag for edge list, based on diffference of u ids
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_flag, no_of_edges);

	ClearArray<<< grid_vertexlen, threads_vertexlen, 0>>>((unsigned int*)d_vertex, no_of_vertices);
	MakeFlagForVertexList<<< grid_edgelen, threads_edgelen, 0>>>(d_pick_array, d_edge_flag, no_of_edges);

	MakeVertexList<<< grid_edgelen, threads_edgelen, 0>>>(d_vertex, d_pick_array, d_edge_flag, no_of_edges);
	
	CUDA_SAFE_CALL( cudaMemcpy( &no_of_edges, d_edge_list_size, sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL( cudaMemcpy( &no_of_vertices, d_vertex_list_size, sizeof(int), cudaMemcpyDeviceToHost));

}



////////////////////////////////////////////////
//Free All memory from Host and Device
////////////////////////////////////////////////
void FreeMem()
{
	free(h_edge);
	free(h_vertex);
	free(h_weight);
	free(h_output_MST_test);
	free(h_vertex_split_rank_test);
	free(h_edge_rank_test);
	CUDA_SAFE_CALL(cudaFree(d_edge));
	CUDA_SAFE_CALL(cudaFree(d_vertex));
	CUDA_SAFE_CALL(cudaFree(d_weight));
	CUDA_SAFE_CALL(cudaFree(d_segmented_min_scan_input));
	CUDA_SAFE_CALL(cudaFree(d_segmented_min_scan_output));
	CUDA_SAFE_CALL(cudaFree(d_edge_flag));
	CUDA_SAFE_CALL(cudaFree(d_pick_array));
	CUDA_SAFE_CALL(cudaFree(d_successor));
	CUDA_SAFE_CALL(cudaFree(d_successor_copy));
	CUDA_SAFE_CALL(cudaFree(d_output_MST));
	CUDA_SAFE_CALL(cudaFree(d_succchange));
	CUDA_SAFE_CALL(cudaFree(d_vertex_split));
	CUDA_SAFE_CALL(cudaFree(d_vertex_split_scratchmem));
	CUDA_SAFE_CALL(cudaFree(d_vertex_flag));
	CUDA_SAFE_CALL(cudaFree(d_new_supervertexIDs));
	CUDA_SAFE_CALL(cudaFree(d_old_uIDs));
	CUDA_SAFE_CALL(cudaFree(d_edge_split_scratchmem));
	CUDA_SAFE_CALL(cudaFree(d_size));
	CUDA_SAFE_CALL(cudaFree(d_edge_mapping));
	CUDA_SAFE_CALL(cudaFree(d_edge_mapping_copy));
	CUDA_SAFE_CALL(cudaFree(d_edge_list_size));
	CUDA_SAFE_CALL(cudaFree(d_vertex_list_size));
	CUDA_SAFE_CALL(cudaFree(d_vertex_split_rank));
	CUDA_SAFE_CALL(cudaFree(d_vertex_rank_scratchmem));
	CUDA_SAFE_CALL(cudaFree(d_edge_rank));
	CUDA_SAFE_CALL(cudaFree(d_edge_rank_scratchmem));
	CUDA_SAFE_CALL(cudaFree(d_appended_uvw));
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	if(argc<2)
	{
	printf("Specify an Input Graph\n");
	exit(1);
	}
	ReadGraph(argv[1]);
	Init();

	unsigned int	timer;
	CUT_SAFE_CALL( cutCreateTimer( &timer));	
	CUT_SAFE_CALL( cutStartTimer( timer));
	//Perform Our MST algorhtm
	do
	{
	    HPGMST();
	}
	while(no_of_vertices>1);
	CUT_SAFE_CALL( cutStopTimer( timer));
	printf("\n=================== Time taken To perform MST :: %3.3f ms===================\n",cutGetTimerValue(timer));


	//Copy the Final MST array to the CPU memory, a 1 at the index means that edge was selected in the MST, 0 otherwise.
	//It should be noted that each edge has an opposite edge also, out of whcih only one is selected in this output.
	//So total number of 1s in this array must be equal to no_of_vertices_orig-1.
	CUDA_SAFE_CALL( cudaMemcpy( h_output_MST_test, d_output_MST, sizeof(unsigned int)*no_of_edges_orig, cudaMemcpyDeviceToHost));
	int k=0;
	int weight=0;
	//printf("\n\nSelected Edges in MST...\n\n");
	for(int i=0;i<no_of_edges_orig;i++)
		if(h_output_MST_test[i]==1)
			{
				//printf("%d %d\n",h_edge[i],h_weight[i]);
				k++;
				weight+=h_weight[i];
			}
	printf("\nNumber of edges in MST, must be=(no_of_vertices-1)): %d,(%d)\nTotal MST weight: %d\n",k, no_of_vertices_orig,weight);
	
	FreeMem();
}

