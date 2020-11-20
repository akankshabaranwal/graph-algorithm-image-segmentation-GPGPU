#include <iostream>

#include "CreateGraph.h"
#include <cuda_runtime_api.h>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::cuda;

int main(int argc, char **argv)
{
    Mat image, output;
    GpuMat dev_image, dev_output;

    image = imread("data/beach.png", IMREAD_COLOR);
    dev_image.upload(image);

    Ptr<Filter> filter = createGaussianFilter(CV_8UC3, CV_8UC3, Size(5, 5), 1.0);
    filter->apply(dev_image, dev_output);

    //Graph parameters
    int numVertices = image.rows*image.cols;
    int numEdges= image.rows*image.cols*8;

    //Convert image to graph
    int *VertexList;
    edge *EdgeList;

    cudaMallocManaged(&VertexList,numVertices*sizeof(int));
    cudaMallocManaged(&EdgeList,numEdges*sizeof(edge));

    dim3 threadsPerBlock(32,32);
    dim3 numBlocks(image.rows/threadsPerBlock.x,image.cols/threadsPerBlock.y);
    ImagetoGraph<<<numBlocks,threadsPerBlock>>>(dev_image, VertexList, EdgeList);
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        printf("CUDA Error in ImagetoGraph function call: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    dev_output.download(output);
    imshow("Source Image", image);
    imshow("After Blur (CUDA)", output);

    waitKey();

    return 0;
}
