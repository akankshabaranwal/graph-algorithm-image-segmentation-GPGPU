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

__global__ void ClearFlagArray(int *flag, int numElements);
__global__ void MarkSegments(int *flag, int *VertexList,int numElements);
void SegmentedReduction(CudaContext& context, int32_t *flag, int32_t *a, int32_t *Out, int numElements, int numSegs);
__global__ void FindSuccessorArray(int32_t *Successor, int32_t *VertexList, int32_t *Out, int numSegments);
__global__ void RemoveCycles(int32_t *Successor, int numVertices);
void PropagateRepresentativeVertices(int *Successor, int numVertices);

//Maybe these functions are not required??
void SortedSplit(int *Representative, int *Vertex, int *Successor, int *Flag2, int numVertices);
__global__ void CreateSuperVertexArray(int *SuperVertexId, int *Vertex, int *Flag2, int numSegments);
__global__ void RemoveSelfEdges(int *SuperVertexId, int *Vertex, int *Flag2, int numSegments);
void CreateUid(int *uid, int *flag, int numElements);
__global__ void RemoveSelfEdges(int *BitEdgeList, int numEdges, int *uid, int *SuperVertexId);
__global__ void CreateUVWArray(int *BitEdgeList, int numEdges, int *uid, int *SuperVertexId, int *UV, int *W);
int SortUVW(int *UV, int *W, int numEdges, int *flag3);
void CreateNewEdgeVertexList(int *newBitEdgeList, int *newVertexList, int *UV, int *W, int *flag3, int new_edge_size, int *flag4);

#endif //FELZENSZWALB_FASTMST_H