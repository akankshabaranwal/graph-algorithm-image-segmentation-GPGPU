//
// Created by akanksha on 28.11.20.
//
#include "FastMST.h"

using namespace mgpu;

////////////////////////////////////////////////////////////////////////////////
// Scan
//https://moderngpu.github.io/faq.html
void DemoSegReduceCsr(CudaContext& context, int *flag, int *a, int *Out) {
    const int numElements = 10;
    const int numSegs = 3;

    printf("Input values:\n");
    for(int i =0;i<10;i++)
        printf("%d, ", a[i]);
    printf("\n");

    printf("Flag values:\n");
    for(int i =0;i<3;i++)
        printf("%d, ", flag[i]);
    printf("\n");

    SegReduceCsr(a, flag, numElements, numSegs,false, Out,(int)INT_MAX, mgpu::minimum<int>(), context);
    cudaDeviceSynchronize();

    printf("Output values:\n");

    for(int i =0;i<3;i++)
        printf("%d, ", Out[i]);
    printf("\n");
}