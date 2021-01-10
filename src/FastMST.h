//
// Created by akanksha on 28.11.20.
//

#ifndef FELZENSZWALB_FASTMST_H
#define FELZENSZWALB_FASTMST_H

#include "CreateGraph.h"

using namespace cv::cuda;
using namespace cv;

__global__ void SetBitEdgeListArray( uint64_t *W,int numElements);


__global__ void ClearFlagArray(uint32_t *flag, int numElements);
__global__ void MarkSegments(uint32_t *flag, uint32_t *VertexList,int numElements);

__global__ void MakeIndexArray( uint32_t *VertexList, uint64_t *tempArray2, uint64_t *tempArray, int numVertices, int numEdges);
__global__ void CreateNWEArray(uint32_t *NWE, uint64_t *MinSegmentedList, int numVertices);
__global__ void FindSuccessorArray(uint32_t *Successor, uint32_t *OnlyEdge, uint32_t *NWE, int numSegments);
__global__ void RemoveCycles(uint32_t *Successor, int numVertices);

void PropagateRepresentativeVertices(uint32_t *Successor, int numVertices);
//void PropagateRepresentativeVertices(uint32_t *Successor, int numVertices, int numBlocks, int numThreads);
__global__ void appendSuccessorArray(uint32_t *Representative, uint32_t *VertexIds, uint32_t *Successor, int numVertices);
__global__ void CreateFlag2Array(uint32_t *Representative, uint32_t *Flag2, int numSegments);
__global__ void CreateSuperVertexArray(uint32_t *SuperVertexId, uint32_t *Vertex, uint32_t *Flag2, int numSegments);
__global__ void RemoveSelfEdges(uint32_t *OnlyEdge, int numEdges, uint32_t *uid, uint32_t *SuperVertexId);
__global__ void CreateUVWArray( uint32_t *OnlyEdge,  uint64_t *OnlyWeight,int numEdges, uint32_t *uid, uint32_t *SuperVertexId, uint64_t *UVW);
__global__ void CreateFlag3Array(uint64_t *UVW, int numEdges, uint32_t *flag3, int *MinMaxScanArray);
__global__ void ResetCompactLocationsArray(uint32_t *compactLocations, uint32_t numEdges);
__global__ void CreateNewEdgeList( uint32_t *compactLocations, uint32_t *newOnlyE, uint64_t *newOnlyW, uint64_t *UVW, uint32_t *flag3, uint32_t new_edge_size, int *new_E_size, int *new_V_size, uint32_t *expanded_u);

__global__ void CreateFlag4Array(uint32_t *expanded_u, uint32_t *Flag4, int numEdges);
__global__ void CreateNewVertexList(uint32_t *VertexList, uint32_t *Flag4, int new_E_size, uint32_t *expanded_u);

#endif //FELZENSZWALB_FASTMST_H