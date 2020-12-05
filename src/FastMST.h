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

//__global__ void FindMinWeightedEdge(cv::cuda::GpuMat Image, int *VertexList, edge *EdgeList, int pitch, int channels);
void SegmentedReduction(CudaContext& context, int32_t *flag, int32_t *a, int32_t *Out, int numElements, int numSegs);

#endif //FELZENSZWALB_FASTMST_H
