//
// Created by akanksha on 28.11.20.
//
#include "FastMST.h"
#include <thrust/execution_policy.h>


////////////////////////////////////////////////////////////////////////////////
// Scan
//https://moderngpu.github.io/faq.html

__global__ void SetBitEdgeListArray( uint64_t *W,uint numElements)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    uint64_t tmp_Wt;
    for (uint32_t idx = tidx; idx < numElements; idx += num_threads)
    {
        tmp_Wt = static_cast<uint64_t> (W[idx]);
        W[idx] = (tmp_Wt << 32) | idx;
    }
}

__global__ void ClearFlagArray(uint32_t *flag, int numElements)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    for (uint32_t idx = tidx; idx < numElements; idx += num_threads)
    {
        flag[idx] = uint32_t(0);
    }
}

__global__ void MarkSegments(uint32_t *flag, uint32_t *VertexList,int numElements)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    for (uint32_t idx = tidx; idx < numElements; idx += num_threads)
    {
        flag[VertexList[idx]] = 1;
    }
    if(tidx==0) flag[VertexList[tidx]]=0;
}

__global__ void MakeIndexArray( uint32_t *VertexList, uint64_t *tempArray2, uint64_t *tempArray, int numVertices, int numEdges)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    for (uint32_t idx = tidx+1; idx < numVertices; idx += num_threads)
    {
        tempArray[idx-1] = tempArray2[VertexList[idx]-1];
    }
    if(tidx=numVertices-1)
        tempArray[tidx] = tempArray2[numEdges-1];
}

__global__ void CreateNWEArray(uint32_t *NWE, uint64_t *MinSegmentedList, int numVertices)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    for (uint32_t idx = tidx; idx < numVertices; idx += num_threads)
    {
        NWE[idx] = MinSegmentedList[idx]&0x00000000FFFFFFFF;
    }
}

__global__ void FindSuccessorArray(uint32_t *Successor,  uint32_t *OnlyEdge, uint32_t *NWE, int numVertices)
{
    uint32_t min_edge_index;
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    for (uint32_t idx = tidx; idx < numVertices; idx += num_threads)
    {   min_edge_index = NWE[idx];
        Successor[idx] = OnlyEdge[min_edge_index];
    }
}

__global__ void RemoveCycles(uint32_t *Successor, int numVertices)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    uint32_t successor_2;

    for (uint32_t idx = tidx; idx < numVertices; idx += num_threads)
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

void PropagateRepresentativeVertices(uint32_t *Successor, int numVertices)
{
    bool change;
    change = true;
    uint32_t successor, successor_2;

    while(change)
    {
        change = false;
        for(uint vertex=0; vertex<numVertices;vertex++)
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

__global__ void appendSuccessorArray(uint32_t *Representative, uint32_t *VertexIds, uint32_t *Successor, int numVertices)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    for (uint32_t idx = tidx; idx < numVertices; idx += num_threads)
    {
        Representative[idx] = Successor[idx];
        VertexIds[idx] = idx;
    }
}

__global__ void CreateFlag2Array(uint32_t *Representative, uint32_t *Flag2, int numSegments)
{
    uint32_t L0, L1;
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    for (uint32_t idx = tidx+1; idx < numSegments; idx += num_threads)
    {
        L0  = Representative[idx-1];
        L1 = Representative[idx];
        if(L1!= L0)
            Flag2[idx]=1;
        else
            Flag2[idx]=0;
    }
    if(tidx==0)
    {
        Flag2[tidx]=0;
    }
}

__global__ void CreateSuperVertexArray(uint32_t *SuperVertexId, uint32_t *VertexIds, uint32_t *Flag2, int numVertices){
    uint32_t vertex, supervertex_id;
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    for (uint32_t idx = tidx; idx < numVertices; idx += num_threads)
    {
        vertex = VertexIds[idx];
        supervertex_id = Flag2[idx];
        SuperVertexId[vertex] = supervertex_id;
    }
}

//11 Removing self edges
__global__ void RemoveSelfEdges(uint32_t *OnlyEdge, int numEdges, uint32_t *uid, uint32_t *SuperVertexId)
{
    uint32_t supervertexid_u, supervertexid_v, id_u, id_v;
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    for (uint32_t idx = tidx; idx < numEdges; idx += num_threads)
    {
        id_u = uid[idx];
        supervertexid_u = SuperVertexId[id_u];

        id_v = OnlyEdge[idx];
        supervertexid_v = SuperVertexId[id_v];

        if(supervertexid_u == supervertexid_v)
        {
            OnlyEdge[idx] = UINT32_MAX; //Marking edge to remove it
        }
    }
}

//12 Removing duplicate edges
__global__ void CreateUVWArray( uint32_t *OnlyEdge, uint64_t *OnlyWeight, int numEdges, uint32_t *uid, uint32_t *SuperVertexId, uint64_t *UVW )
{
    uint32_t id_u, id_v, edge_weight;
    uint64_t supervertexid_u, supervertexid_v;
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    for (uint32_t idx = tidx; idx < numEdges; idx += num_threads)
    {
        id_u = uid[idx];
        id_v =  OnlyEdge[idx];
        edge_weight = OnlyWeight[idx]>>32;
        if(id_v != UINT32_MAX) //Check if the edge is marked using the criteria from before
        {
            supervertexid_u = SuperVertexId[id_u];
            supervertexid_v = SuperVertexId[id_v];
            UVW[idx] = (supervertexid_u<<38)|(supervertexid_v<<12)|edge_weight;
        }
        else
        {
            UVW[idx] = UINT64_MAX;
        }
    }
}

__global__ void CreateFlag3Array(uint64_t *UVW, int numEdges, uint32_t *flag3, int *MinMaxScanArray)
{
    uint32_t prev_supervertexid_u, prev_supervertexid_v, supervertexid_u, supervertexid_v;
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    MinMaxScanArray[tidx]=1;
    for (uint32_t idx = tidx+1; idx < numEdges; idx += num_threads)
    {
        prev_supervertexid_u = UVW[idx-1]>>38;
        prev_supervertexid_v = (UVW[idx-1]>>12) &0x0000000003FFFFFF;

        supervertexid_u = UVW[idx]>>38;
        supervertexid_v = (UVW[idx]>>12) &0x0000000003FFFFFF;

        flag3[idx] = 0;
        MinMaxScanArray[idx]=1;
        if((supervertexid_u!=67108863) and (supervertexid_v!=4095) and (supervertexid_u!=-1) and (supervertexid_v!=-1) and (supervertexid_u!=4095) and (supervertexid_v!=67108863)and (UVW[idx]!=UINT64_MAX))
            {
            if((prev_supervertexid_u !=supervertexid_u) || (prev_supervertexid_v!=supervertexid_v))
            {
                flag3[idx] = 1;
            }
            else
            {
                flag3[idx] = 0;
            }
                MinMaxScanArray[idx]=idx+1;
            }
    }
    if(tidx==0)flag3[tidx]=1;

}

__global__ void ResetCompactLocationsArray(uint32_t *compactLocations, uint32_t numEdges)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    for (uint32_t idx = tidx; idx < numEdges; idx += num_threads)
    {
        compactLocations[idx] = compactLocations[idx]-1;
    }
}

__global__ void CreateNewEdgeList( uint32_t *compactLocations, uint32_t *newOnlyE, uint64_t *newOnlyW, uint64_t *UVW, uint32_t *flag3, uint32_t new_edge_size, int *new_E_size, int *new_V_size, uint32_t *expanded_u)
{
    uint32_t supervertexid_u, supervertexid_v;
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    uint64_t edgeWeight;
    uint32_t newLocation;

    for (uint32_t idx = tidx; idx < new_edge_size; idx += num_threads)
    {
        new_E_size[idx] = 0;
        new_V_size[idx] = 0;
        if(flag3[idx])
        {
            supervertexid_u =UVW[idx]>>38;
            supervertexid_v =((UVW[idx]>>12)&0x000003FFFFFF);
            edgeWeight = (UVW[idx]&0x00000000FFF);
            newLocation = compactLocations[idx];
            if((supervertexid_u!=67108863) and (supervertexid_v!=4095) and (supervertexid_u!=-1) and (supervertexid_v!=-1) and (supervertexid_u!=4095) and (supervertexid_v!=67108863)and (UVW[idx]!=UINT64_MAX))
            {
                newOnlyE[newLocation] = supervertexid_v;
                newOnlyW[newLocation] = (edgeWeight<<32) | newLocation;
                expanded_u[newLocation] = supervertexid_u;
                new_E_size[idx] = newLocation +1;
                new_V_size[idx] = supervertexid_v +1;
            }
        }
    }
}

__global__ void CreateFlag4Array(uint32_t *expanded_u, uint32_t *Flag4, int numEdges)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    for (uint32_t idx = tidx+1; idx < numEdges; idx += num_threads)
    {
        Flag4[idx]=0;
        if(expanded_u[idx-1]!=expanded_u[idx])
        {
            Flag4[idx]=1;
        }
    }
    if(tidx==0)Flag4[tidx]=1;
}

__global__ void CreateNewVertexList(uint32_t *VertexList, uint32_t *Flag4, int new_E_size, uint32_t *expanded_u)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    uint32_t id_u;

    for (uint32_t idx = tidx; idx < new_E_size; idx += num_threads)
    {
     if(Flag4[idx] == 1)
     {
         id_u = expanded_u[idx];
         VertexList[id_u] = idx;
     }
    }
}
