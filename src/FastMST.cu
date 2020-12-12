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
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<numElements)
    {
        flag[id] = 0;
    }
}

__global__ void MarkSegments(int *flag, int *VertexList,int numElements)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<numElements)
    {
        flag[VertexList[id]] = 1;
    }
}
/*
__global__ void CreateNWEArray(int32_t *NWE, int32_t *Out, int numSegments)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < numSegments)
    {          NWE[id] = Out[id] % (2 << 15);
    }
}*/

void SegmentedReduction(CudaContext& context, int32_t *flag, int32_t *a, int32_t *Out, int numElements, int numSegs)
{
    //Segmented min scan
    SegReduceCsr(a, flag, numElements, numSegs, false, Out,(int32_t)INT_MAX, mgpu::minimum<int32_t>(),context);
    //cudaDeviceSynchronize();

    //Create NWE array with the index of each minimum edge
    //int numthreads = 1024;
    //int numBlock = numSegs/numthreads;
    //CreateNWEArray<<<numBlock,numthreads>>>(NWE, Out, numSegs);
    /*cudaError_t err = cudaGetLastError();
    err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        printf("CUDA Error in CreateNWEArray function call: %s\n", cudaGetErrorString(err));
    }*/
    //cudaDeviceSynchronize();
}

__global__ void FindSuccessorArray(int32_t *Successor, int32_t *VertexList, int32_t *Out, int numVertices)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t min_edge_index;

    if (id < numVertices)
    {   min_edge_index = VertexList[id];
        Successor[id] = Out[min_edge_index]%(2<<15);
    }
}

__global__ void RemoveCycles(int32_t *Successor, int numVertices)
{
    int vertex = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t successor_2;
    if(vertex<numVertices)
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
/*
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
}*/

void PropagateRepresentativeVertices(int *Successor, int numVertices)
{
    bool change =true;
    while(change)
    {
        change = false;
        //int numthreads = 1024;
        //int numBlock = numSegments/numthreads;
        //TODO: Is it worth to make this parallel? Repeat copy of 'change' between host and device??
        //Try some scan to figure out value of change array after every kernel call?
        int32_t successor, successor_2;
        for(int i=0; i<numVertices;i++)
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
void SortedSplit(int *Representative, int *Vertex, int *Successor, int *Flag2, int numVertices)
{
    int numthreads = 1024;
    int numBlock = numVertices/numthreads;
    appendSuccessorArray<<<numBlock,numthreads>>>(Representative, Vertex, Successor, numVertices);
    cudaError_t err = cudaGetLastError();
    err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        printf("CUDA Error in appendSuccessorArray function call: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    thrust::sort_by_key(thrust::host, Representative, Representative + numVertices, Vertex);

    CreateFlagArray<<<numBlock,numthreads>>>(Representative, Vertex, Flag2, numVertices);
    //Scan to assign new vertex ids. Use exclusive scan. Run exclusive scan on the flag array
    thrust::inclusive_scan(Flag2, Flag2 + numVertices, Flag2, thrust::plus<int>());
}

//TODO: The array names need to be verified
__global__ void CreateSuperVertexArray(int *SuperVertexId, int *Vertex, int *Flag2, int numSegments){
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
            BitEdgeList[idx] = INT_MAX; //Marking edge to remove it
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
        if(BitEdgeList[idx] != INT_MAX) //Check if the edge is marked using the criteria from before
        {
            supervertexid_u = SuperVertexId[id_u];
            supervertexid_v = SuperVertexId[id_v];
            UV[idx] = supervertexid_u*(2<<15) + supervertexid_v; //TODO: Check if the UV here needs to be 64bit??
            W[idx] = edge_weight;
        }
        else
        {
            UV[idx] = INT_MAX;
            W[idx] = INT_MAX; //TODO: Need to replace the -1 with INT_MAX
        }
    }
}

//FIXME: Add code for clearing of flag array before every place it has been used
int SortUVW(int *UV, int *W, int numEdges, int *flag3)
{
    //12.2
    thrust::sort_by_key(thrust::host, UV, UV + numEdges, W);
    //12.3
    //Initialize F3 array
    int new_edge_size = numEdges;
    int32_t prev_supervertexid_u, prev_supervertexid_v, supervertexid_u, supervertexid_v;

    // TODO: Check how to replace the min(newEdges part so that this can be parallelized.
    for(int i=1;i<numEdges;i++)
    {
        prev_supervertexid_u = UV[i-1]>>15; //TODO: Check if this needs to be 15 or 16
        prev_supervertexid_v = UV[i-1] %(2<<15);

        supervertexid_u = UV[i]>>15;
        supervertexid_v = UV[i]%(2<<15);

        if((supervertexid_u!=INT_MAX) and (supervertexid_v!=INT_MAX))
        {
            if((prev_supervertexid_u !=supervertexid_v) || (prev_supervertexid_v!=supervertexid_v))
            {
                flag3[i] = 1;
            }
            else
            {
                flag3[i] = 0;
                new_edge_size = min(new_edge_size, i); // Basically we are setting new_edge_size to the last index wherever the value is not max.
                //FIXME: For this to work the while marking the edges you need to set it as infinity instead of -1.
            }
        }
    }
    return new_edge_size;
}
/*
__global__ void createInParallel(int *newBitEdgeList, int *newVertexList, int *compact_locations, int new_edge_size, int *flag3, int *UV, int *W)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int32_t supervertex_id_u, supervertex_id_v, edge_weight;
    int new_location;

    if(idx < new_edge_size)
    {
        if(flag3[idx]){
            supervertex_id_u = UV[idx]>>15;
            supervertex_id_v = UV[idx]%(2<<15);
            edge_weight = W[idx];
            new_location = compact_locations [idx];
            if()
        }
    }
}*/

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
void CreateNewEdgeVertexList(int *newBitEdgeList, int *newVertexList, int *UV, int *W, int *compact_locations, int *flag3, int new_edge_size)
{
    //Check if this can be parallelized? can we move the min (new_edge_size) part to somewhere before?
    int32_t supervertex_id_u, supervertex_id_v;

    //13.1 Create the Compact new edge list
    //Scan of flag3 array and store it in compact_locations.
    //TODO: You can rename flag3 to compact_locations, so this way you dont need 2 arrays
    //FIXME: Same fixme as above for replacing -1 with INT_MAX
    thrust::inclusive_scan(flag3, flag3 + new_edge_size, compact_locations, thrust::plus<int>());
    //TODO: Check if New_E_Size and New V Size would be <= Number of 1s in the flag array??
    //FIXME: Need to allocate memory for newBitEdgeList and newVertexList appropriately.
    //TODO: Check if we can just override the variables here and dont need to repeatedly create BitEdgeList and vertexList
    int new_E_size = 0;
    int new_V_size =0;
    //expand_u is not the same as newVertexList. newVertexList will get created later

    int *expand_u;
    cudaMallocManaged(&expand_u, new_edge_size * sizeof(int32_t));
    // FIXME: Need to find a way to get newEdgeListLength and newVertexListLength in parallel

    int edge_weight;
    int new_location;

    new_V_size=0;
    new_E_size = 0;
    for(int i=0;i < new_edge_size; i++)
    {
        if(flag3[i])
        {
            supervertex_id_u = UV[i]>>15;
            supervertex_id_v = UV[i]%(2<<15);
            edge_weight = W[i];
            new_location = compact_locations [i];
            //FIXME: Replace -1 with infinity
            if((supervertex_id_v!= INT_MAX)&&(supervertex_id_u!=INT_MAX))
            {
                newBitEdgeList[i] = edge_weight*(2<<15) + supervertex_id_v;
                expand_u[i] = supervertex_id_u;
                new_edge_size = max(new_location+1, new_edge_size);
                new_V_size = max(supervertex_id_v+1, new_V_size);
            }
        }
    }

    //Skipping step 13.3 as not required in C
    //14. Building a new vertex list
    int *flag4; //Same as F4. New flag for creating vertex list. Assigning the new ids.
    cudaMallocManaged(&flag4, new_E_size * sizeof(int));
    if(new_E_size>0)
        flag4[0] =1;
    //Create the flag array in parallel
    int numthreads = 1024;
    int numBlock = new_E_size/numthreads;
    CreateFlag4Array<<<numBlock, numthreads>>>(expand_u, flag4, new_E_size);
    //Can this flag4 array creation be skipped?? Can we directly use the index while creating the expand_u array?
    CreateNewVertexList<<<numBlock, numthreads>>>(newVertexList, flag4, new_E_size, expand_u);
}