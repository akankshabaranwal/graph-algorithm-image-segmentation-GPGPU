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

__global__ void ClearFlagArray(uint *flag, int numElements);
__global__ void MarkSegments(uint *flag, uint32_t *VertexList,int numElements);
void SegmentedReduction(CudaContext& context, uint32_t *VertexList, uint64_t *BitEdgeList, uint64_t *MinSegmentedList, int numEdges, int numVertices);
__global__ void CreateNWEArray(uint32_t *NWE, uint64_t *MinSegmentedList, int numVertices);
__global__ void FindSuccessorArray(uint32_t *Successor, uint64_t *BitEdgeList, uint32_t *NWE, int numSegments);
__global__ void RemoveCycles(uint32_t *Successor, int numVertices);

void PropagateRepresentativeVertices(uint32_t *Successor, int numVertices);
__global__ void appendSuccessorArray(uint32_t *Representative, uint32_t *VertexIds, uint32_t *Successor, int numVertices);
__global__ void CreateFlag2Array(uint32_t *Representative, uint *Flag2, int numSegments);
__global__ void CreateSuperVertexArray(uint32_t *SuperVertexId, uint32_t *Vertex, uint *Flag2, int numSegments);
void CreateUid(uint32_t *uid, uint *flag, int numElements);
__global__ void RemoveSelfEdges(uint32_t *OnlyEdge, int numEdges, uint32_t *uid, uint32_t *SuperVertexId);
__global__ void CreateUVWArray(uint64_t *BitEdgeList, uint32_t *OnlyEdge, int numEdges, uint32_t *uid, uint32_t *SuperVertexId, uint64_t *UV, uint32_t *W);
__global__ void CreateFlag3Array(uint64_t *UV, uint32_t *W, int numEdges, uint *flag3, uint32_t *MinMaxScanArray);
__global__ void ResetCompactLocationsArray(uint32_t *compactLocations, uint32_t numEdges);
__global__ void CreateNewEdgeList(uint64_t *BitEdgeList, uint32_t *compactLocations, uint32_t *newOnlyE, uint64_t *newOnlyW, uint64_t *UV, uint32_t *W, uint *flag3, uint32_t new_edge_size, uint32_t *new_E_size, uint32_t *new_V_size, uint32_t *expanded_u);

__global__ void CreateFlag4Array(uint32_t *expanded_u, uint *Flag4, int numEdges);
__global__ void CreateNewVertexList(uint32_t *newVertexList, uint *Flag4, int new_E_size, uint32_t *expanded_u);

#endif //FELZENSZWALB_FASTMST_H