//
// Created by akanksha on 28.11.20.
//
#include "FastMST.h"
#include <thrust/execution_policy.h>

using namespace mgpu;

////////////////////////////////////////////////////////////////////////////////
// Scan
//https://moderngpu.github.io/faq.html

__global__ void SetBitEdgeListArray(uint64_t *BitEdgeList, uint32_t *OnlyEdge, uint64_t *W,uint numElements)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    uint64_t tmp_Wt;
    for (uint idx = tidx; idx < numElements; idx += num_threads)
    {
        tmp_Wt = static_cast<uint64_t> (W[idx]);
        BitEdgeList[idx] = (tmp_Wt<<32)|OnlyEdge[idx];
    }
}

__global__ void SetOnlyWeightArray(uint64_t *BitEdgeList,uint64_t *OnlyWeight, uint numElements)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    uint64_t tmp_Wt;
    for (uint idx = tidx; idx < numElements; idx += num_threads)
    {
        tmp_Wt =BitEdgeList[idx]>>32;
        OnlyWeight[idx] = (tmp_Wt << 32) | idx;
    }
}

__global__ void ClearFlagArray(uint *flag, int numElements)
{
    uint tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;

    for (uint idx = tidx; idx < numElements; idx += num_threads)
    {
        flag[idx] = 0;
    }
}

__global__ void MarkSegments(uint *flag, uint32_t *VertexList,int numElements)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    for (uint32_t idx = tidx; idx < numElements; idx += num_threads)
    {
        flag[VertexList[idx]] = 1;
    }
}
__global__ void IncrementVertexList(uint32_t *VertexList,int numElements)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    for (uint32_t idx = tidx; idx < numElements; idx += num_threads)
    {
        VertexList[idx] = VertexList[idx] +1;
    }
}

void SegmentedReduction(CudaContext& context, uint32_t *VertexList, uint64_t *BitEdgeList, uint64_t *MinSegmentedList, int numEdges, int numVertices)
{

    SegReduceCsr(BitEdgeList, VertexList, numEdges, numVertices, false, MinSegmentedList,(uint64_t)UINT64_MAX, mgpu::minimum<uint64_t>(),context);
    //InputIt data_global, CsrIt csr_global, int count,int numSegments, bool supportEmpty, OutputIt dest_global, T identity, Op op,CudaContext& context)
}

__global__ void CreateNWEArray(uint32_t *NWE, uint64_t *MinSegmentedList, int numVertices)
{
    uint tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;

    for (uint idx = tidx; idx < numVertices; idx += num_threads)
    {
        NWE[idx] = MinSegmentedList[idx]&0x00000000FFFFFFFF;
    }
}

__global__ void FindSuccessorArray(uint32_t *Successor, uint64_t *BitEdgeList, uint32_t *NWE, int numVertices)
{
    uint32_t min_edge_index;
    uint tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;

    for (uint idx = tidx; idx < numVertices; idx += num_threads)
    {   min_edge_index = NWE[idx];
        Successor[idx] = BitEdgeList[min_edge_index]&0x00000000FFFFFFFF;
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

__global__ void CreateFlag2Array(uint32_t *Representative, uint *Flag2, int numSegments)
{
    Flag2[0]=0;
    uint32_t L0, L1;
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

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

__global__ void CreateSuperVertexArray(uint32_t *SuperVertexId, uint32_t *VertexIds, uint *Flag2, int numVertices){
    // Find supervertex id. Create a supervertex array for the original vertex ids
    uint32_t vertex, supervertex_id;

    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    //printf("numThreads: %d", num_threads);
    for (uint32_t idx = tidx; idx < numVertices; idx += num_threads)
    {
        vertex = VertexIds[idx];
        supervertex_id = Flag2[idx];
        SuperVertexId[vertex] = supervertex_id;
    }
}

//10.2
void CreateUid(uint32_t *uid, uint *flag, int numElements)
{
    flag[0]=0;
    thrust::inclusive_scan(flag, flag + numElements, uid, thrust::plus<int>());
}

//11 Removing self edges
__global__ void RemoveSelfEdges(uint32_t *OnlyEdge, int numEdges, uint32_t *uid, uint32_t *SuperVertexId)
{  // int idx = blockIdx.x*blockDim.x+threadIdx.x;
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
//Instead of UVW array create separate U, V, W array.
__global__ void CreateUVWArray(uint64_t *BitEdgeList, uint32_t *OnlyEdge, int numEdges, uint32_t *uid, uint32_t *SuperVertexId, uint64_t *UVW )
{
    uint32_t id_u, id_v, edge_weight;
    uint64_t supervertexid_u, supervertexid_v;
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    for (uint32_t idx = tidx; idx < numEdges; idx += num_threads)
    {
        id_u = uid[idx];
        id_v =  OnlyEdge[idx];
        edge_weight = BitEdgeList[idx]>>32;
        if(id_v != UINT32_MAX) //Check if the edge is marked using the criteria from before
        {
            supervertexid_u = SuperVertexId[id_u];
            supervertexid_v = SuperVertexId[id_v];
            //UV[idx] = supervertexid_u<<32 |supervertexid_v;
            //W[idx] = edge_weight;
            UVW[idx] = (supervertexid_u<<44)|(supervertexid_v<<22)|edge_weight;
        }
        else
        {
         // UV[idx] = UINT64_MAX;
         // W[idx] = edge_weight;
            UVW[idx] = UINT64_MAX;
        }
    }
}

__global__ void CreateFlag3Array(uint64_t *UVW, int numEdges, uint *flag3, uint32_t *MinMaxScanArray)
{
    uint32_t prev_supervertexid_u, prev_supervertexid_v, supervertexid_u, supervertexid_v;
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    MinMaxScanArray[tidx]=1;
    for (uint32_t idx = tidx+1; idx < numEdges; idx += num_threads)
    {
        //prev_supervertexid_u = UV[idx-1]>>32;
        //prev_supervertexid_v = UV[idx-1] &0x00000000FFFFFFFF;

        //supervertexid_u = UV[idx]>>32;
        //supervertexid_v = UV[idx]&0x00000000FFFFFFFF;

        prev_supervertexid_u = UVW[idx-1]>>44;
        prev_supervertexid_v = (UVW[idx-1]>>22) &0x000003FFFFF;

        supervertexid_u = UVW[idx]>>44;
        supervertexid_v = (UVW[idx]>>22) &0x000003FFFFF;

        flag3[idx] = 0;
        MinMaxScanArray[idx]=1;
        if((supervertexid_u!=1048575) and (supervertexid_v!=4194303) and (supervertexid_u!=-1) and (supervertexid_v!=-1)and UVW[idx]!=UINT64_MAX)
        {
            if((prev_supervertexid_u !=supervertexid_u) || (prev_supervertexid_v!=supervertexid_v))
            {
                flag3[idx] = 1;
            }
            else
            {
                flag3[idx] = 0;
                MinMaxScanArray[idx]=idx+1;
            }
        }
    }
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

__global__ void CreateNewEdgeList(uint64_t *BitEdgeList, uint32_t *compactLocations, uint32_t *newOnlyE, uint64_t *newOnlyW, uint64_t *UVW, uint *flag3, uint32_t new_edge_size, uint32_t *new_E_size, uint32_t *new_V_size, uint32_t *expanded_u)
{
    uint32_t supervertexid_u, supervertexid_v;
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    uint64_t edgeWeight;
    uint32_t newLocation;

    for (uint idx = tidx; idx < new_edge_size; idx += num_threads)
    {
        new_E_size[idx] = 0;
        new_V_size[idx] = 0;
        if(flag3[idx])
        {
            //supervertexid_u =UV[idx]>>32;
            //supervertexid_v =UV[idx]&0x00000000FFFFFFFF;
            //edgeWeight = W[idx];
            supervertexid_u =UVW[idx]>>44;
            supervertexid_v =((UVW[idx]>>22)&0x000003FFFFF);
            edgeWeight = (UVW[idx]&0x000000FFFFF);
            newLocation = compactLocations[idx];
            if((supervertexid_u!=4194303) and (supervertexid_v!=1048575) and (supervertexid_u!=-1) and (supervertexid_v!=-1) and (supervertexid_u!=1048575) and (supervertexid_v!=4194303))
            {
                newOnlyE[newLocation] = supervertexid_v;
                newOnlyW[newLocation] = (edgeWeight<<32) | newLocation;

                BitEdgeList[newLocation] = (edgeWeight <<32) |supervertexid_v;
                expanded_u[newLocation] = supervertexid_u;
                new_E_size[idx] = newLocation +1;
                new_V_size[idx] = supervertexid_v +1;
            }
        }
    }
}

__global__ void CreateFlag4Array(uint32_t *expanded_u, uint *Flag4, int numEdges)
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
}

__global__ void CreateNewVertexList(uint32_t *VertexList, uint *Flag4, int new_E_size, uint32_t *expanded_u)
{
    uint32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;
    uint32_t id_u;

    for (uint idx = tidx; idx < new_E_size; idx += num_threads)
    {
     if(Flag4[idx] == 1)
     {
         id_u = expanded_u[idx];
         VertexList[id_u] = idx;
     }
    }
}
