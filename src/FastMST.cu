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

    //for (int i = 0; i < numSegs; i++)
    //{
        //NWE[i] = Out[i] % (2 << 15);
     //   printf("%d, ", NWE[i]);
        //if((NWE[i]<0)||(NWE[i]>59999))
        //{
        //    printf("Indexing Error for Out %d Flag %d!!\n", Out[i], flag[i]);
        //}
    //}
}

__global__ void FindSuccessorArray(int32_t *Successor, int32_t *NWE, int numSegments)
{
 /* Iterate through the NWE array from SegmentedReduction
  * Create a Successor array in parallel based on the vertex id of the minimum edge weight that was selected
  */
    // Get our global thread ID
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
        //TODO: Is it worth to make this parallel? Repeat copy of change between host and device??
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

//https://thrust.github.io/doc/group__sorting_gabe038d6107f7c824cf74120500ef45ea.html#gabe038d6107f7c824cf74120500ef45ea
void SortedSplit(int *Representative, int *Vertex, int *Successor, int numSegments)
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

    //printf("Before sort:\n");
    //numSegments = 60000;
    //for(int i =0; i<numSegments; i++)
    //    printf(" %d,%d ;", Representative[i], Vertex[i]);
    //printf("\n");
    thrust::sort_by_key(thrust::host, Representative, Representative + numSegments, Vertex);
    //printf("After sort:\n");
    //for(int i =0; i<numSegments; i++)
    //    printf(" %d,%d ;", Representative[i], Vertex[i]);
    //printf("\n");
}