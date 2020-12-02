//
// Created by akanksha on 28.11.20.
//
#include "FastMST.h"

using namespace mgpu;

////////////////////////////////////////////////////////////////////////////////
// Scan
//https://moderngpu.github.io/faq.html
void DemoSegReduceCsr(CudaContext& context, int *flag, int *a, int *Out, int numElements, int numSegs) {

    printf("Input values:\n");
    for(int i =0;i<numElements;i++)
        printf("%d, ", a[i]);
    printf("\n");

    printf("Flag values:\n");
    for(int i =0;i<numSegs;i++)
        printf("%d, ", flag[i]);
    printf("\n");

    SegReduceCsr(a, flag, numElements, numSegs,false, Out,(int)INT_MAX, mgpu::minimum<int>(), context);
    cudaDeviceSynchronize();

    printf("Output values:\n");

    for(int i =0;i<numSegs;i++)
        printf("%d, Weight: %d\n", Out[i], int(Out[i]>>16));
    printf("\n");
}