#include <iostream>

#include "CreateGraph.h"
#include <cuda_runtime_api.h>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>
#include "moderngpu.cuh"		// Include all MGPU kernels.
#include "FastMST.h"

using namespace cv;
using namespace cv::cuda;

using namespace mgpu;

int main(int argc, char **argv)
{
    Mat image, output;
    GpuMat dev_image, dev_output;

    image = imread("data/beach.png", IMREAD_COLOR);

    //TODO: Add checker for image size depending on the bits decided for representing edge weight and vertex index
    dev_image.upload(image);

    Ptr<Filter> filter = createGaussianFilter(CV_8UC3, CV_8UC3, Size(5, 5), 1.0);
    filter->apply(dev_image, dev_output);

    //Graph parameters
    int numVertices = image.rows*image.cols;
    int numEdges= image.rows*image.cols*8;

    //Convert image to graph
    int *VertexList, *BitEdgeList;
    edge *EdgeList;

    cudaMallocManaged(&VertexList,numVertices*sizeof(int));
    cudaMallocManaged(&EdgeList,numEdges*sizeof(edge));
    cudaMallocManaged(&BitEdgeList,numEdges*sizeof(edge));

    dim3 threadsPerBlock(32,32);
    int BlockX = image.rows/threadsPerBlock.x;
    int BlockY = image.cols/threadsPerBlock.y;
    dim3 numBlocks(BlockX,BlockY);

    ImagetoGraph<<<numBlocks,threadsPerBlock>>>(dev_output, VertexList, EdgeList, BitEdgeList, dev_output.step, dev_output.channels());
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        printf("CUDA Error in ImagetoGraph function call: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    printf("INFO: Checking the edge list and bit edge list\n");
    //TODO: Fix bug in edge list creation
    for(int i =0;i<numEdges;i++)
    {
        printf("%d %f %d\n", EdgeList[i].Vertex, EdgeList[i].Weight, BitEdgeList[i]);
    }
    //dev_output.download(output);
    //imshow("Source Image", image);
    //imshow("After Blur (CUDA)", output);

    //waitKey();

    ContextPtr context = CreateCudaDevice(argc, argv, true);
    //    int a[10] = {2,3,1,4,5,6,0,8,4,10};
//    int flag[10] = {0,0,0,1,0,0,1,0,1,0};
//    int Out[4];

    int *a;
    int *flag;
    int *Out;
    cudaMallocManaged(&a,10*sizeof(int));
    cudaMallocManaged(&flag,3*sizeof(int));
    cudaMallocManaged(&Out,3*sizeof(int));

    a[0] = 2;
    a[1] =3;
    a[2] = 10;
    a[3] = 3;
    a[4] = 5;
    a[5] = 6;
    a[6] = 100;
    a[7] = 8;
    a[8] = 4;
    a[9] = 6;

    flag[0]=0;
    flag[1]=3;
    flag[2]=7;

    //AB: Marksegments is not required because flag array is already like that

    //flag[2] = 0;
    //flag[3] = 1;
    //flag[4]=0;
    //flag[5]=0;
    //flag[6]=0;
    //flag[7] =1;
    //flag[8] =0;
    //flag[9] = 0;

    DemoSegReduceCsr(*context, flag, a, Out);

    //for(int i=0;i<4;i++)
    //    printf("%d ,", Out[i]);
    //https://moderngpu.github.io/segreduce.html

    //Print the pitch information
 //   cudaDeviceProp devProp;
 //   cudaGetDeviceProperties(&devProp, 0);
 //   printf("Maximum memory pitch:%lu\n",  devProp.memPitch);

    return 0;
}
