//
// Created by akanksha on 28.11.20.
//

#ifndef FELZENSZWALB_FASTMST_H
#define FELZENSZWALB_FASTMST_H

#include "CreateGraph.h"
#include "moderngpu.cuh"		// Include all MGPU kernels.

using namespace cv::cuda;
using namespace cv;
using namespace mgpu;

#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
    {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

__global__ void ClearFlagArray(int *flag, int numElements);
__global__ void MarkSegments(int *flag, int *VertexList,int numElements);
void SegmentedReduction(CudaContext& context, int32_t *VertexList, int32_t *BitEdgeList, int32_t *MinSegmentedList, int numEdges, int numVertices);
__global__ void CreateNWEArray(int32_t *NWE, int32_t *MinSegmentedList, int numVertices);
__global__ void FindSuccessorArray(int32_t *Successor, int32_t *VertexList, int32_t *MinSegmentedList, int numSegments);
__global__ void RemoveCycles(int32_t *Successor, int numVertices);

void PropagateRepresentativeVertices(int32_t *Successor, int numVertices);
__global__ void appendSuccessorArray(int32_t *Representative, int32_t *VertexIds, int32_t *Successor, int numVertices);
void SortedSplit(int32_t *Representative, int32_t *VertexIds, int32_t *Successor, int *Flag2, int numVertices);
__global__ void CreateSuperVertexArray(int *SuperVertexId, int *Vertex, int *Flag2, int numSegments);
void CreateUid(int *uid, int *flag, int numElements);
__global__ void RemoveSelfEdges(int *BitEdgeList, int numEdges, int *uid, int *SuperVertexId);
__global__ void CreateUVWArray(int *BitEdgeList, int numEdges, int *uid, int *SuperVertexId, int *UV, int *W);
int SortUVW(int32_t *UV, int *W, int numEdges, int *flag3);
int CreateNewEdgeVertexList(int *newBitEdgeList, int *newVertexList, int *UV, int *W, int *flag3, int new_edge_size, int *flag4);

#endif //FELZENSZWALB_FASTMST_H