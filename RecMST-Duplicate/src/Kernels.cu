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

  Kernels for MST implementation, by Pawan Harish.
 ************************************************************************************/

#ifndef _KERNELS_H_
#define _KERNELS_H_

#define MOVEBITS 22 						// Amount of bits in X for vertex ID
#define NO_OF_BITS_TO_SPLIT_ON 32			// Amount of bits for L split (32 bits one vertex, 32 other)
#define NO_OF_BITS_MOVED_FOR_VERTEX_IDS 24
#define NO_OF_BITS_TO_SPLIT_ON_UVW 64
#define MAX_THREADS_PER_BLOCK 1024 // IMPORTANT TO SET CORRECTLY
#define INF 10000000


////////////////////////////////////////////////////////////////////////////////////////////
// Append the Weight And Vertex ID into segmented min scan input array, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////////////////
__global__ void AppendKernel_1(unsigned int *d_segmented_min_scan_input, unsigned int *d_weight, unsigned int *d_edges, unsigned int no_of_edges) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges) {
		unsigned int val=d_weight[tid];
		val=val<<MOVEBITS;
		val=val|d_edges[tid];
		d_segmented_min_scan_input[tid]=val;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Make the flag for Input to the segmented min scan, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////
__global__ void ClearArray(unsigned int *d_array, unsigned int size) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<size) {
		d_array[tid]=0;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Make the flag for Input to the segmented min scan, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeFlag_3(unsigned int *d_edge_flag, unsigned int *d_vertex, unsigned int no_of_vertices) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		unsigned int pointingvertex = d_vertex[tid];
		d_edge_flag[pointingvertex]=1;
	}
}


////////////////////////////////////////////////////////////////////////////////
// Make the Successor array, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeSucessorArray(unsigned int *d_successor, unsigned int *d_vertex, unsigned int *d_segmented_min_scan_output, unsigned int no_of_vertices, unsigned int no_of_edges) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		unsigned int end; // Result values always stored at end of each segment
		if(tid<no_of_vertices-1) {
			end = d_vertex[tid+1]-1; // Get end of my segment
		} else {
			end = no_of_edges-1; // Last segment: end = last edge
		}
		unsigned int mask = pow(2.0,MOVEBITS)-1; // Mask to extract vertex ID MWOE
		d_successor[tid] = d_segmented_min_scan_output[end]&mask; // Get vertex part of each (weight|to_vertex_id) element
	}
}

////////////////////////////////////////////////////////////////////////////////
// Remove Cycles Using Successor array, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void RemoveCycles(unsigned int *d_successor, unsigned int no_of_vertices) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x; // TID: vertex ID
	if(tid<no_of_vertices) {
		unsigned int succ = d_successor[tid];
		unsigned int nextsucc = d_successor[succ];
		if(tid == nextsucc) { //Found a Cycle
			//Give the minimum one its own value, breaking the cycle and setting the Representative Vertices
			if(tid < succ) {
				d_successor[tid]=tid;
			} else {
				d_successor[succ]=succ;
			}
		}
	}
}


__global__ void SuccToCopy(unsigned int *d_successor, unsigned int *d_successor_copy, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		d_successor_copy[tid] = d_successor[tid];
	}
}

////////////////////////////////////////////////////////////////////////////////
// Propagate Representative IDs by setting S(u)=S(S(u)), Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void PropagateRepresentativeID(unsigned int *d_successor, unsigned int *d_successor_copy, bool *d_succchange, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		unsigned int succ = d_successor[tid];
		unsigned int newsucc = d_successor[succ];
		if(succ!=newsucc) { //Execution goes on
			d_successor_copy[tid] = newsucc; //cannot have input and output in the same array!!!!!
			*d_succchange=true;
		}
	}
}

__global__ void CopyToSucc(unsigned int *d_successor, unsigned int *d_successor_copy, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
		d_successor[tid] = d_successor_copy[tid];
}


////////////////////////////////////////////////////////////////////////////////
// Append Vertex IDs with SuperVertex IDs, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void AppendVertexIDsForSplit(unsigned long long int *d_vertex_sort, unsigned int *d_successor, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
	{
		unsigned long long int val;
		val = d_successor[tid]; // representative
		val = val<<NO_OF_BITS_TO_SPLIT_ON;
		val |= tid; // u
		d_vertex_sort[tid]=val; // (representative, vertex)
	}
}

////////////////////////////////////////////////////////////////////////////////
// Mark New SupervertexID per vertex, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeSuperVertexIDPerVertex(unsigned int *d_new_supervertexIDs, unsigned long long int *d_vertex_split, unsigned int *d_vertex_flag, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		unsigned long long int mask = pow(2.0,NO_OF_BITS_TO_SPLIT_ON)-1;
		unsigned long long int vertexid = d_vertex_split[tid]&mask;
		d_vertex_flag[vertexid] = d_new_supervertexIDs[tid];
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copy New SupervertexID per vertex, resolving read after write inconsistancies, Runs for Vertex Length  // IMPORTANT! RAW
////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void CopySuperVertexIDPerVertex(unsigned int *d_new_supervertexIDs, unsigned int *d_vertex_flag, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		d_new_supervertexIDs[tid] = d_vertex_flag[tid];
	}
}


////////////////////////////////////////////////////////////////////////////////
// Make flag for Scan, assigning new ids to supervertices, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeFlagForScan(unsigned int *d_vertex_flag, unsigned long long int *d_split_input, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		if(tid>0) {
			unsigned long long int val = d_split_input[tid-1];
			unsigned long long int supervertexid_prev  = val>>NO_OF_BITS_TO_SPLIT_ON;
			val = d_split_input[tid];
			unsigned long long int supervertexid  = val>>NO_OF_BITS_TO_SPLIT_ON;
			if(supervertexid_prev!=supervertexid) {
				d_vertex_flag[tid]=1;
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// Make flag to assign old vertex ids, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeFlagForUIds(unsigned int *d_edge_flag, unsigned int *d_vertex, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		if(tid>0) {
			unsigned int pointingvertex = d_vertex[tid];
			d_edge_flag[pointingvertex]=1;
		}
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copy Edge Array to somewhere to resolve read after write inconsistancies, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void CopyEdgeArray(unsigned int *d_edge, unsigned int *d_edge_mapping_copy, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges)
		d_edge_mapping_copy[tid] = d_edge[tid];
}

////////////////////////////////////////////////////////////////////////////////
// Remove self edges based on new supervertex ids, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////
__global__ void RemoveSelfEdges(unsigned int *d_edge, unsigned int *d_old_uIDs, unsigned int *d_new_supervertexIDs, unsigned int *d_edge_mapping_copy, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges) {
		unsigned int uid = d_old_uIDs[tid];
		unsigned int vid = d_edge[tid];
		unsigned int usuperid = d_new_supervertexIDs[uid];
		unsigned int vsuperid = d_new_supervertexIDs[vid];
		if(usuperid == vsuperid){
			d_edge_mapping_copy[tid]=INF; //Nullify the edge if both vertices have same supervertex id, do not use the same array for output
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copy Edge Array Back, resolving read after write inconsistancies, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void CopyEdgeArrayBack(unsigned int *d_edge, unsigned int *d_edge_mapping_copy, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges)
		d_edge[tid]=d_edge_mapping_copy[tid];
}


////////////////////////////////////////////////////////////////////////////////
// Append U,V,W for duplicate edge removal, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////
__global__ void AppendForDuplicateEdgeRemoval(unsigned long long int *d_appended_uvw, unsigned int *d_edge, unsigned int *d_old_uIDs, unsigned int *d_weight, unsigned int *d_new_supervertexIDs, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges) {
		unsigned long long int val;
		unsigned int u,v,superuid=INF,supervid=INF;
		u = d_old_uIDs[tid];
		v = d_edge[tid];

		if (v == INF) { // TODO: maybe useful. else (u, INF, w)
			u = INF;
		}

		if(u!=INF && v!=INF) {
			superuid = d_new_supervertexIDs[u];
			supervid = d_new_supervertexIDs[v];
		}
		val = superuid;
		val = val<<NO_OF_BITS_MOVED_FOR_VERTEX_IDS;
		val |= supervid;
		val = val<<(64-(NO_OF_BITS_MOVED_FOR_VERTEX_IDS+NO_OF_BITS_MOVED_FOR_VERTEX_IDS));
		val |= d_weight[tid];
		d_appended_uvw[tid]=val;
	}
}


////////////////////////////////////////////////////////////////////////////////
// Mark the starting edge for each uv combination, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MarkEdgesUV(unsigned int *d_edge_flag, unsigned long long int *d_appended_uvw, unsigned int *d_size, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges) {
		if(tid>0) {
			unsigned long long int test = INF;
			test = test << NO_OF_BITS_MOVED_FOR_VERTEX_IDS;
			test |=INF;
			unsigned long long int test1 = d_appended_uvw[tid]>>(64-(NO_OF_BITS_MOVED_FOR_VERTEX_IDS+NO_OF_BITS_MOVED_FOR_VERTEX_IDS)); // uv[i]
			unsigned long long int test2 = d_appended_uvw[tid-1]>>(64-(NO_OF_BITS_MOVED_FOR_VERTEX_IDS+NO_OF_BITS_MOVED_FOR_VERTEX_IDS)); // uv[i-1]

			if(test1>test2) {
				d_edge_flag[tid]=1;
			}

			if(test1 == test) { // TODO: might be different if change line 334. Not sure if correct now either
				atomicMin(d_size,tid); //also to know the last element in the array, i.e. the size of new edge list
			}
		} else {
			d_edge_flag[tid]=1;
		}
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////
// Compact the edgelist and weight list, keep a mapping for each edge, Runs for d_size Length
///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void CompactEdgeList(unsigned int *d_edge, unsigned int *d_weight, 
								unsigned int *d_old_uIDs, unsigned int *d_edge_flag, unsigned long long int *d_appended_uvw,
								unsigned int *d_pick_array, unsigned int *d_size, 
								unsigned int *d_edge_list_size, unsigned int *d_vertex_list_size)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<*d_size) {
		if(	d_edge_flag[tid]==1) {
			unsigned long long int UVW = d_appended_uvw[tid];
			unsigned int writepos = d_old_uIDs[tid];
			unsigned long long int mask = pow(2.0,64-(NO_OF_BITS_MOVED_FOR_VERTEX_IDS+NO_OF_BITS_MOVED_FOR_VERTEX_IDS))-1;
			unsigned long long int w  = UVW&mask;
			unsigned long long int test = UVW>>(64-(NO_OF_BITS_MOVED_FOR_VERTEX_IDS+NO_OF_BITS_MOVED_FOR_VERTEX_IDS));
			unsigned long long int mask2 = pow(2.0,NO_OF_BITS_MOVED_FOR_VERTEX_IDS)-1;
			unsigned long long int v = test&mask2;
			unsigned long long int u = test>>NO_OF_BITS_MOVED_FOR_VERTEX_IDS;
			if(u!=INF && v!=INF) {
				//Copy the edge_mapping into a temporary array, used to resolve read after write inconsistancies
				d_pick_array[writepos]=u; // reusing this to store u's
				d_edge[writepos] = v;
				d_weight[writepos] = w;
				//max writepos will give the new edge list size
				atomicMax(d_edge_list_size,(writepos+1));
				atomicMax(d_vertex_list_size,(v+1));
				// Orig: atomicMax(d_vertex_list_size,(u+1)); //how can max(v) be > max(u), error!!!!! TODO check this whole thing
			}
		}		
	}
}

////////////////////////////////////////////////////////////////////////////////
//Copy the temporary array to the actual mapping array, Runs for Edge length
////////////////////////////////////////////////////////////////////////////////
__global__ void CopyEdgeMap(unsigned int *d_edge_mapping, unsigned int *d_edge_mapping_copy, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges)
		d_edge_mapping[tid] = d_edge_mapping_copy[tid]; 
}

////////////////////////////////////////////////////////////////////////////////
//Make Flag for Vertex List Compaction, Runs for Edge length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeFlagForVertexList(unsigned int *d_pick_array, unsigned int *d_edge_flag, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges) {
		if(tid>0) {
			if(d_pick_array[tid] != d_pick_array[tid-1]) { //This line may be causing problems TODO: maybe != such as in python code but should be fine. Change back to > for orig
				d_edge_flag[tid]=1;
			}
		} else {
			d_edge_flag[tid]=1;
			//atomicMax(d_edge_list_size,(tid));
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//Vertex List Compaction, Runs for Edge length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeVertexList(unsigned int *d_vertex, unsigned int *d_pick_array, unsigned int *d_edge_flag, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges) {
		if(d_edge_flag[tid]==1) {
			unsigned int writepos=d_pick_array[tid]; //get the u value
			d_vertex[writepos]=tid; //write the index to the u'th value in the array to create the vertex list
			//atomicMax(d_vertex_list_size,(writepos+1));
		}
	}
}


#endif // #ifndef _KERNELS_H_