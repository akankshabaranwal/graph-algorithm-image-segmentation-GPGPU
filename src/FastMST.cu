//
// Created by akanksha on 28.11.20.
//
#include "FastMST.h"

using namespace mgpu;

////////////////////////////////////////////////////////////////////////////////
// Scan
//https://moderngpu.github.io/faq.html
void SegmentedReduction(CudaContext& context, int32_t *flag, int32_t *a, int32_t *Out, int numElements, int numSegs)
{
    printf("Input values:\n");
    for (int i = 0; i < 100; i++)
        printf("%d, ", a[i]);
    printf("\n");

    printf("Flag values:\n");
    for (int i = 0; i < 100; i++)
        printf("%d, ", flag[i]);
    printf("\n");

    SegReduceCsr(a, flag, numElements, numSegs, false, Out, (int)INT_MAX, mgpu::minimum<int>(), context);
    cudaDeviceSynchronize();

    printf("Output values:\n");

    for (int i = 0; i < 100; i++)
    {
        //printf(" %d, ", Out[i]);
        printf("%d %d\n", Out[i] % (2 << 16), Out[i] >> 16);
    }
    printf("\n");
}