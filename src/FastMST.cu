//
// Created by akanksha on 28.11.20.
//
#include "FastMST.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

using namespace mgpu;

////////////////////////////////////////////////////////////////////////////////
// Scan
//https://moderngpu.github.io/faq.html

__global__ void MarkSegments(int *flag, int *VertexList,int numElements)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<numElements)
    {
        flag[VertexList[id]] = 1;
    }
}

__global__ void CreateNWEArray(int32_t *NWE, int32_t *Out, int numSegments)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < numSegments)
    {          NWE[id] = Out[id] % (2 << 15);
    }
}

void SegmentedReduction(CudaContext& context, int32_t *flag, int32_t *a, int32_t *Out, int32_t *NWE, int numElements, int numSegs)
{
    SegReduceCsr(a, flag, numElements, numSegs, false, Out,(int32_t)INT_MAX, mgpu::minimum<int32_t>(),context);
    cudaDeviceSynchronize();

    //Create NWE array with the index of each minimum edge
    int numthreads = 1024;
    int numBlock = numSegs/numthreads;
    CreateNWEArray<<<numBlock,numthreads>>>(NWE, Out, numSegs);
    cudaError_t err = cudaGetLastError();
    err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        printf("CUDA Error in CreateNWEArray function call: %s\n", cudaGetErrorString(err));
    }
    //cudaDeviceSynchronize();

}

__global__ void FindSuccessorArray(int32_t *Successor, int32_t *NWE, int numSegments)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t min_edge_index;

    if (id < numSegments)
    {   min_edge_index = NWE[id]; //TODO: Check if this is correct. This will eliminate passing the Out array to this kernel
        Successor[id] = NWE[min_edge_index];
    }
}

__global__ void RemoveCycles(int32_t *Successor, int numSegments)
{
    int vertex = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t successor_2;
    if(vertex<numSegments)
    {
        successor_2 = Successor[Successor[vertex]];
        if(vertex == successor_2) //Cycle detected
        {
            if(vertex < successor_2)
                Successor[vertex] = vertex;
            else
            {
                Successor[Successor[vertex]] = Successor[vertex];
            }
        }
    }
}

__global__ void PropagateParallel(int32_t *Successor, int numSegments, bool *change)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t successor_2, successor;
    successor = Successor[id];
    successor_2 = Successor[successor];
    if (id < numSegments)
    {   if(successor!=successor_2)
        {
            *change = true;
            Successor[id] = successor_2;
        }
    }
    //TODO: How to return boolean change??
}

void PropagateRepresentativeVertices(int *Successor, int numSegments)
{
    bool change =true;
    while(change)
    {
        change = false;
        int numthreads = 1024;
        int numBlock = numSegments/numthreads;
        //TODO: Is it worth to make this parallel? Repeat copy of 'change' between host and device??
        //PropagateParallel<<<numBlock,numthreads>>>(Successor, numSegments, change);
        int32_t successor, successor_2;
        for(int i=0; i<numSegments;i++)
        {
            successor = Successor[i];
            successor_2 = Successor[successor];
            if(successor!=successor_2)
            {
                change=true;
                Successor[i] = successor_2;
            }
        }
    }
}

__global__ void appendSuccessorArray(int *Representative, int *Vertex, int *Successor, int numSegments)
{
    int vertex = blockIdx.x*blockDim.x+threadIdx.x;
    if(vertex<numSegments)
    {
        Representative[vertex] = Successor[vertex];
        Vertex[vertex] = vertex;
    }
}

__global__ void CreateFlagArray(int *Representative, int *Vertex, int *Flag2, int numSegments)
{
    int vertex = blockIdx.x*blockDim.x+threadIdx.x;
    if((vertex<numSegments)&&(vertex>0))
    {
        if(Representative[vertex] != Representative[vertex-1])
            Flag2[vertex]=1;
        else
            Flag2[vertex]=0;
    }
}

//https://thrust.github.io/doc/group__sorting_gabe038d6107f7c824cf74120500ef45ea.html#gabe038d6107f7c824cf74120500ef45ea
void SortedSplit(int *Representative, int *Vertex, int *Successor, int *Flag2, int numSegments)
{
    int numthreads = 1024;
    int numBlock = numSegments/numthreads;
    appendSuccessorArray<<<numBlock,numthreads>>>(Representative, Vertex, Successor, numSegments);
    cudaError_t err = cudaGetLastError();
    err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        printf("CUDA Error in appendSuccessorArray function call: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    thrust::sort_by_key(thrust::host, Representative, Representative + numSegments, Vertex);

    CreateFlagArray<<<numBlock,numthreads>>>(Representative, Vertex, Flag2, numSegments);
    //Scan to assign new vertex ids. Use exclusive scan. Run exclusive scan on the flag array
    thrust::inclusive_scan(Flag2, Flag2 + numSegments, Flag2, thrust::plus<int>());
}

//TODO: The array names need to be verified
__global__ void RemoveSelfEdges(int *SuperVertexId, int *Vertex, int *Flag2, int numSegments){
    // Find supervertex id. Create a supervertex array for the original vertex ids
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t vertex, supervertex_id;
    if(idx < numSegments)
    {
        vertex = Vertex[idx];
        supervertex_id = Flag2[idx];
        SuperVertexId[vertex] = Vertex[idx];
    }
}

//10.2
void CreateUid(int *uid, int *flag, int numElements)
{
    thrust::inclusive_scan(flag, flag + numElements, uid, thrust::plus<int>());
}

//11 Removing self edges
__global__ void RemoveSelfEdges(int *BitEdgeList, int numEdges, int *uid, int *SuperVertexId)
{   int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t supervertexid_u, supervertexid_v, id_u, id_v;
    if(idx<numEdges)
    {
        id_u = uid[idx];
        supervertexid_u = SuperVertexId[id_u];

        id_v = BitEdgeList[idx]% (2 << 15);
        supervertexid_v = SuperVertexId[id_v];

        if(supervertexid_u == supervertexid_v)
        {
            BitEdgeList[idx] = -1; //Marking edge to remove it
        }
    }
}


//12 Removing duplicate edges
//Instead of UVW array create separate U, V, W array.
__global__ void CreateUVWArray(int *BitEdgeList, int numEdges, int *uid, int *SuperVertexId, int *UV, int *W)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x; //Index for accessing Edge
    int32_t id_u, id_v, edge_weight;
    int32_t supervertexid_u, supervertexid_v;
    if(idx < numEdges)
    {
        id_u = uid[idx];
        id_v = BitEdgeList[idx]>>15; //TODO: Check if this is correct
        edge_weight = BitEdgeList[idx]% (2 << 15);//TODO: Check if we can use the NWE array?
        if(id_v != -1) //Check if the edge is marked using the criteria from before
        {
            supervertexid_u = SuperVertexId[id_u];
            supervertexid_v = SuperVertexId[id_v];
            UV[idx] = supervertexid_u*(2<<15) + supervertexid_v; //TODO: Check if the UV here needs to be 64bit??
            W[idx] = edge_weight;
        }
        else
        {
            UV[idx] = -1;
            W[idx] = -1; //TODO: Need to replace the -1 with INT_MAX
        }
    }
}

//12.2
void SortUVW(int *UV, int *W, int numEdges)
{
    thrust::sort_by_key(thrust::host, UV, UV + numEdges, W);
    //12.3
    //Initialize F3 array

}

//Create new edge list and vertex list
void CreateNewEdgeVertexList(int *newBitEdgeList, int *newVertexList, int *U, int *V, int *W, int numnewEdges, int numnewVertices)
{
    //Check if this can be parallelized? can we move the min (new_edge_size) part to somewhere before?
    int32_t supervertex_id_u, supervertex_id_v;
    for(int i=0;i<numnewEdges;i++)
    {

    }
}
