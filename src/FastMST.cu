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

__global__ void ClearFlagArray(int *flag, int numElements)
{
    unsigned int tidx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int num_threads = gridDim.x * blockDim.x;

    for (uint idx = tidx; idx < numElements; idx += num_threads)
    {
        flag[idx] = 0;
    }
}

__global__ void MarkSegments(int *flag, int *VertexList,int numElements)
{
    int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t num_threads = gridDim.x * blockDim.x;
    for (uint idx = tidx; idx < numElements; idx += num_threads)
    {
        flag[VertexList[idx]] = 1;
    }
}

void SegmentedReduction(CudaContext& context, int32_t *VertexList, int32_t *BitEdgeList, int32_t *MinSegmentedList, int numEdges, int numVertices)
{
    SegReduceCsr(BitEdgeList, VertexList, numEdges, numVertices, false, MinSegmentedList,(int32_t)INT_MAX, mgpu::minimum<int32_t>(),context);
}

__global__ void CreateNWEArray(int32_t *NWE, int32_t *MinSegmentedList, int numVertices)
{
    unsigned int tidx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int num_threads = gridDim.x * blockDim.x;

    for (uint idx = tidx; idx < numVertices; idx += num_threads)
    {
        NWE[idx] = MinSegmentedList[idx]%(2<<15);
    }
}

__global__ void FindSuccessorArray(int32_t *Successor, int32_t *BitEdgeList, int32_t *NWE, int numVertices)
{
    int32_t min_edge_index;
    unsigned int tidx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int num_threads = gridDim.x * blockDim.x;

    for (uint idx = tidx; idx < numVertices; idx += num_threads)
    {   min_edge_index = NWE[idx];
        Successor[idx] = BitEdgeList[min_edge_index]%(2<<15);
    }
}

__global__ void RemoveCycles(int32_t *Successor, int numVertices)
{
    int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t num_threads = gridDim.x * blockDim.x;
    int32_t successor_2;

    for (int32_t idx = tidx; idx < numVertices; idx += num_threads)
    {
        successor_2 = Successor[Successor[idx]];
        if(idx == successor_2) //Cycle detected
        {
            if(idx < Successor[idx])
                Successor[idx] = idx;
            else
            {
                Successor[Successor[idx]] = Successor[idx];
            }
        }
    }
}

void PropagateRepresentativeVertices(int32_t *Successor, int numVertices)
{
    bool change;
    change = true;
    int32_t successor, successor_2;

    while(change)
    {
        change = false;
        for(int vertex=0; vertex<numVertices;vertex++)
        {
            successor = Successor[vertex];
            successor_2 = Successor[successor];
            if(successor!=successor_2)
            {
                change=true;
                Successor[vertex] = successor_2;
            }
        }
    }
}

__global__ void appendSuccessorArray(int32_t *Representative, int32_t *VertexIds, int32_t *Successor, int numVertices)
{
    int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t num_threads = gridDim.x * blockDim.x;
    for (uint idx = tidx; idx < numVertices; idx += num_threads)
    {
        Representative[idx] = Successor[idx];
        VertexIds[idx] = idx;
    }
}

__global__ void CreateFlag2Array(int32_t *Representative, int *Flag2, int numSegments)
{
    Flag2[0]=0;
    int32_t L0, L1;
    int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t num_threads = gridDim.x * blockDim.x;

    for (uint idx = tidx+1; idx < numSegments; idx += num_threads)
    {
        L0  = Representative[idx-1];
        L1 = Representative[idx];
        if(L1!= L0)
            Flag2[idx]=1;
        else
            Flag2[idx]=0;
    }
}

//TODO: The array names need to be verified
__global__ void CreateSuperVertexArray(int *SuperVertexId, int *VertexIds, int *Flag2, int numVertices){
    // Find supervertex id. Create a supervertex array for the original vertex ids
    int32_t vertex, supervertex_id;

    int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t num_threads = gridDim.x * blockDim.x;
    //printf("numThreads: %d", num_threads);
    for (uint idx = tidx; idx < numVertices; idx += num_threads)
    {
        vertex = VertexIds[idx];
        supervertex_id = Flag2[idx];
        SuperVertexId[vertex] = supervertex_id;
    }
}

//10.2
void CreateUid(int *uid, int *flag, int numElements)
{
    flag[0]=0;
    thrust::inclusive_scan(flag, flag + numElements, uid, thrust::plus<int>());
}

//11 Removing self edges
__global__ void RemoveSelfEdges(int *OnlyEdge, int numEdges, int *uid, int *SuperVertexId)
{  // int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t supervertexid_u, supervertexid_v, id_u, id_v;

    int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t num_threads = gridDim.x * blockDim.x;

    for (uint idx = tidx; idx < numEdges; idx += num_threads)
    {
        id_u = uid[idx];
        supervertexid_u = SuperVertexId[id_u];

        id_v = OnlyEdge[idx];
        supervertexid_v = SuperVertexId[id_v];

        if(supervertexid_u == supervertexid_v)
        {
            OnlyEdge[idx] = INT_MAX; //Marking edge to remove it
        }
    }
}

//12 Removing duplicate edges
//Instead of UVW array create separate U, V, W array.
__global__ void CreateUVWArray(int32_t *BitEdgeList, int32_t *OnlyEdge, int numEdges, int *uid, int32_t *SuperVertexId, int32_t *UV, int *W)
{
    //int idx = blockIdx.x*blockDim.x+threadIdx.x; //Index for accessing Edge
    int32_t id_u, id_v, edge_weight;
    int32_t supervertexid_u, supervertexid_v;
    int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t num_threads = gridDim.x * blockDim.x;

    for (uint idx = tidx; idx < numEdges; idx += num_threads)
    {
        id_u = uid[idx];
        id_v =  OnlyEdge[idx];
        edge_weight = BitEdgeList[idx]>>16;
        if(id_v != INT_MAX) //Check if the edge is marked using the criteria from before
        {
            supervertexid_u = SuperVertexId[id_u];
            supervertexid_v = SuperVertexId[id_v];
            UV[idx] = supervertexid_u*(2<<15) + supervertexid_v;
            W[idx] = edge_weight;
        }
        else
        {
            UV[idx] = INT_MAX;
            W[idx] = edge_weight; //TODO: Need to replace the -1 with INT_MAX
        }
    }
}

__global__ void CreateFlag3Array(int32_t *UV, int32_t *W, int numEdges, int *flag3, int *MinMaxScanArray)
{

    int32_t prev_supervertexid_u, prev_supervertexid_v, supervertexid_u, supervertexid_v;
    int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t num_threads = gridDim.x * blockDim.x;
    MinMaxScanArray[tidx]=0;
    for (uint idx = tidx+1; idx < numEdges; idx += num_threads)
    {
        prev_supervertexid_u = UV[idx-1]>>16; //TODO: Check if this needs to be 15 or 16
        prev_supervertexid_v = UV[idx-1] %(2<<15);

        supervertexid_u = UV[idx]>>16;
        supervertexid_v = UV[idx]%(2<<15);
        flag3[idx] = 0;
        MinMaxScanArray[idx]=0;
        if((supervertexid_u!=32767) and (supervertexid_v!=65536))
        {
            if((prev_supervertexid_u !=supervertexid_u) || (prev_supervertexid_v!=supervertexid_v))
            {
                flag3[idx] = 1;
            }
            else
            {
                flag3[idx] = 0;
                MinMaxScanArray[idx]=idx;
            }
        }
    }
}

__global__ void ResetCompactLocationsArray(int32_t *compactLocations, int32_t numEdges)
{
    int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t num_threads = gridDim.x * blockDim.x;

    for (uint idx = tidx; idx < numEdges; idx += num_threads)
    {
        compactLocations[idx] = compactLocations[idx]-1;
    }
}

__global__ void CreateNewEdgeList(int *BitEdgeList, int *compactLocations, int *newOnlyE, int*newOnlyW, int *UV, int *W, int *flag3, int new_edge_size, int *new_E_size, int *new_V_size, int *expanded_u)
{
    int32_t supervertexid_u, supervertexid_v;
    int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t num_threads = gridDim.x * blockDim.x;
    int32_t edgeWeight;
    int newLocation;

    for (uint idx = tidx; idx < new_edge_size; idx += num_threads)
    {
        new_E_size[idx] = 0;
        new_V_size[idx] = 0;
        if(flag3[idx])
        {
            supervertexid_u =UV[idx]>>16;
            supervertexid_v =UV[idx]%(2<<15);
            edgeWeight = W[idx];
            //edgeWeight=BitEdgeList[idx]>>16;
            newLocation = compactLocations[idx];
            if((supervertexid_u!=32767) and (supervertexid_v!=65535))
            {
                newOnlyE[newLocation] = supervertexid_v;
                newOnlyW[newLocation] = edgeWeight;
                BitEdgeList[newLocation] = (edgeWeight * (2<<15)) + supervertexid_v;
                expanded_u[newLocation] = supervertexid_u;
                new_E_size[idx] = newLocation +1;
                new_V_size[idx] = supervertexid_v +1;
            }
        }
    }
}

__global__ void CreateFlag4Array(int *expanded_u, int *Flag4, int numEdges)
{

    int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t num_threads = gridDim.x * blockDim.x;

    for (uint idx = tidx+1; idx < numEdges; idx += num_threads)
    {
        Flag4[idx]=0;
        if(expanded_u[idx-1]!=expanded_u[idx])
        {
            Flag4[idx]=1;
        }
    }
}

__global__ void CreateNewVertexList(int *newVertexList, int *Flag4, int new_E_size, int *expanded_u)
{
    int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t num_threads = gridDim.x * blockDim.x;
    int32_t id_u;

    for (uint idx = tidx; idx < new_E_size; idx += num_threads)
    {
     if(Flag4[idx] == 1)
     {
         id_u = expanded_u[idx];
         newVertexList[id_u] = idx;
     }
    }
}