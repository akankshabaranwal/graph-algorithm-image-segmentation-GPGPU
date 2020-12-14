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
    //int id = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tidx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int num_threads = gridDim.x * blockDim.x;

    for (uint idx = tidx; idx < numElements; idx += num_threads)
    {
        flag[idx] = 0;
    }
}

__global__ void MarkSegments(int *flag, int *VertexList,int numElements)
{
    //int id = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t num_threads = gridDim.x * blockDim.x;
    for (uint idx = tidx; idx < numElements; idx += num_threads)
    {
        flag[VertexList[idx]] = 1;
    }
}

void SegmentedReduction(CudaContext& context, int32_t *VertexList, int32_t *BitEdgeList, int32_t *MinSegmentedList, int numEdges, int numVertices)
{
    //Segmented min scan
    SegReduceCsr(BitEdgeList, VertexList, numEdges, numVertices, false, MinSegmentedList,(int32_t)INT_MAX, mgpu::minimum<int32_t>(),context);
}

__global__ void CreateNWEArray(int32_t *NWE, int32_t *MinSegmentedList, int numVertices)
{
    //int id = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t min_edge_index;
    unsigned int tidx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int num_threads = gridDim.x * blockDim.x;

    for (uint idx = tidx; idx < numVertices; idx += num_threads)
    {
        NWE[idx] = MinSegmentedList[idx]%(2<<15);
    }
}

__global__ void FindSuccessorArray(int32_t *Successor, int32_t *BitEdgeList, int32_t *NWE, int numVertices)
{
    //int id = blockIdx.x*blockDim.x+threadIdx.x;
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
    //int vertex = blockIdx.x*blockDim.x+threadIdx.x;
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
    //printf("numThreads: %d", num_threads);
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

void SortedSplit(int32_t *Representative, int32_t *VertexIds, int32_t *Successor, int *Flag2, int numVertices)
{
    int numthreads = 32;
    int numBlock = numVertices/numthreads;
    printf("Calling sorted split number of vertices are:%d ", numVertices);
    //TODO: Make this run on device
    thrust::sort_by_key(thrust::device, Representative, Representative + numVertices, VertexIds);
    CreateFlag2Array<<<numBlock, numthreads>>>(Representative, Flag2, numVertices);
    cudaError_t err = cudaGetLastError();        // Get error code
    err = cudaGetLastError();        // Get error code
    if ( err != cudaSuccess )
    {
        printf("CUDA Error: CreateFlag2 Array%s\n", cudaGetErrorString(err));
        exit(-1);
    }
    //Scan to assign new vertex ids. Use exclusive scan. Run exclusive scan on the flag array
    thrust::inclusive_scan(Flag2, Flag2 + numVertices, Flag2, thrust::plus<int>());
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
        id_v =  OnlyEdge[idx];  //TODO: Check if this is correct
        edge_weight = BitEdgeList[idx]>>16;
        if(id_v != INT_MAX) //Check if the edge is marked using the criteria from before
        {
            supervertexid_u = SuperVertexId[id_u];
            supervertexid_v = SuperVertexId[id_v];
            UV[idx] = supervertexid_u*(2<<15) + supervertexid_v; //TODO: Check if the UV here needs to be 64bit??
            W[idx] = edge_weight;
        }
        else
        {
            UV[idx] = INT_MAX;
            W[idx] = edge_weight; //TODO: Need to replace the -1 with INT_MAX
        }
    }
}

//FIXME: Add code for clearing of flag array before every place it has been used
int SortUVW(int32_t *UV, int32_t *W, int numEdges, int *flag3)
{
    //12.2
    thrust::sort_by_key(thrust::device, UV, UV + numEdges, W);
    cudaDeviceSynchronize();
    printf("\n Printing UVW array after SortUVW: ");
    for(int i = 0; i< numEdges;i++)
    {
        printf("%d %d %d , ", UV[i]>>16, UV[i]%(2<<15), W[i]);
    }
    printf("\n");

    int new_edge_size = numEdges;
    int32_t prev_supervertexid_u, prev_supervertexid_v, supervertexid_u, supervertexid_v;
    // TODO: Check how to replace the min(newEdges part so that this can be parallelized.
    flag3[0]=0;
    for(int i=1;i<numEdges;i++)
    {
        prev_supervertexid_u = UV[i-1]>>16; //TODO: Check if this needs to be 15 or 16
        prev_supervertexid_v = UV[i-1] %(2<<15);

        supervertexid_u = UV[i]>>16;
        supervertexid_v = UV[i]%(2<<15);
        flag3[i] = 0;
        if((supervertexid_u!=INT_MAX) and (supervertexid_v!=INT_MAX) and (UV[i]!=INT_MAX))
        {
            if((prev_supervertexid_u !=supervertexid_v) || (prev_supervertexid_v!=supervertexid_v))
            {
                flag3[i] = 1;
            }
            else
            {
                flag3[i] = 0;
                new_edge_size = min(new_edge_size, i); // Basically we are setting new_edge_size to the last index wherever the value is not max.
            }
        }
    }
    return new_edge_size;
}

//FIXME: The create flag array kernel calls are redundant. Replace them with a single kernel.
__global__ void CreateFlag4Array(int *Representative, int *Flag4, int numSegments)
{
    int vertex = blockIdx.x*blockDim.x+threadIdx.x;
    if((vertex<numSegments)&&(vertex>0))
    {
        if(Representative[vertex] != Representative[vertex-1])
            Flag4[vertex]=1;
        else
            Flag4[vertex]=0;
    }
}

__global__ void CreateNewVertexList(int *newVertexList, int *Flag4, int new_E_size, int *expanded_u)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t id_u;

    if(idx<new_E_size)
    {
     if(Flag4[idx] == 1)
     {
         id_u = expanded_u[idx];
         newVertexList[id_u] = idx;
     }
    }
}

//Create new edge list and vertex list
void CreateNewEdgeVertexList(int *newBitEdgeList, int *newVertexList, int *UV, int *W, int *flag3, int new_edge_size, int *new_E_size, int *new_V_size, int *flag4)
{
    //Check if this can be parallelized? can we move the min (new_edge_size) part to somewhere before?
    int32_t supervertex_id_u, supervertex_id_v;

    //13.1 Create the Compact new edge list
    //Scan of flag3 array and store it in compact_locations.
    int32_t *compact_locations;
    cudaMallocManaged(&compact_locations,new_edge_size*sizeof(int32_t));

    //TODO: You can rename flag3 to compact_locations, so this way you dont need 2 arrays
    thrust::inclusive_scan(flag3, flag3 + new_edge_size, compact_locations, thrust::plus<int>());
    //FIXME: Need to allocate memory for newBitEdgeList and newVertexList appropriately.

    //int new_E_size = 0;
    //int new_V_size =0;

    int *expand_u;
    cudaMallocManaged(&expand_u, new_edge_size * sizeof(int32_t));

    int edge_weight;
    int new_location;

    //new_V_size=0;
    //new_E_size = 0;
    for(int i=0;i < new_edge_size; i++)
    {
        if(flag3[i])
        {
            supervertex_id_u = UV[i]>>15;
            supervertex_id_v = UV[i]%(2<<15);
            edge_weight = W[i];
            new_location = compact_locations [i];

            if((supervertex_id_v!= INT_MAX)&&(supervertex_id_u!=INT_MAX))
            {
                newBitEdgeList[i] = edge_weight*(2<<15) + supervertex_id_v;
                expand_u[i] = supervertex_id_u;
               // new_E_size = max(new_location+1, new_E_size); //Create a MaxESize Array and replace this with MaxScan
               // new_V_size = max(supervertex_id_v+1, new_V_size); //Create a MaxVSize Array and replace this with MaxScan
            }
        }
    }

    //Skipping step 13.3 as not required in C
    //14. Building a new vertex list

    if(new_E_size>0)
        flag4[0] =1;
    //Create the flag array in parallel
    int numthreads = 32;
    int numBlock = new_edge_size/numthreads;

    CreateFlag4Array<<<numBlock, numthreads>>>(expand_u, flag4, new_edge_size);
    cudaError_t err = cudaGetLastError();        // Get error code
    if ( err != cudaSuccess )
    {
        printf("CUDA Error Flag4: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    cudaDeviceSynchronize();

    //Can this flag4 array creation be skipped?? Can we directly use the index while creating the expand_u array?
    CreateNewVertexList<<<numBlock, numthreads>>>(newVertexList, flag4, new_edge_size, expand_u);
    err = cudaGetLastError();        // Get error code
    if ( err != cudaSuccess )
    {
        printf("CUDA Error CreateNewVertexList: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    cudaDeviceSynchronize();

}