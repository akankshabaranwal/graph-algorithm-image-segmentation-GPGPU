#include <iostream>

#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>

#include "mst.h"

int main(int argc, char **argv)
{
    cv::Mat image, output;
    cv::cuda::GpuMat dev_image, dev_output;

    image = imread("data/beach.png", cv::IMREAD_COLOR);
    dev_image.upload(image);

    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(CV_8UC3, CV_8UC3, cv::Size(5, 5), 1.0);
    filter->apply(dev_image, dev_output);

    dev_output.download(output);

    imshow("Source Image", image);
    imshow("After Blur (CUDA)", output);

    char *segmented_img = compute_segments(dev_output.cudaPtr(), image.cols, image.rows);

    imshow("Segmented", cv::Mat(image.rows, image.cols, CV_8UC3, segmented_img));
    cv::waitKey();

    return 0;
}
