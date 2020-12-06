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

void SegmentedReduction(CudaContext& context, int32_t *flag, int32_t *a, int32_t *Out, int32_t *NWE, int numElements, int numSegs);
__global__ void FindSuccessorArray(int32_t *Successor, int32_t *NWE, int numSegments);
__global__ void RemoveCycles(int32_t *Successor, int numSegments);
void PropagateRepresentativeVertices(int *Successor, int numSegments);
void SortedSplit(int *Representative, int *Vertex, int *Successor, int numSegments);


#endif //FELZENSZWALB_FASTMST_H