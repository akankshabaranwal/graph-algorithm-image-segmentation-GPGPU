#include <iostream>

#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>

#include "mst.h"

int main(int argc, char **argv)
{
    cv::Mat image, output;
    cv::cuda::GpuMat dev_image, dev_output;

    image = imread("data/beach.png", cv::IMREAD_COLOR);
    std::cout << "Computing on: " << image.rows << "x" << image.cols << std::endl;
    dev_image.upload(image);

    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(CV_8UC3, CV_8UC3, cv::Size(5, 5), 1.0);
    filter->apply(dev_image, dev_output);

    dev_output.download(output);

    namedWindow("Source Image", cv::WINDOW_NORMAL);
    imshow("Source Image", image);
    namedWindow("After Blur (CUDA)", cv::WINDOW_NORMAL);
    imshow("After Blur (CUDA)", output);

    time_t start, end;
    time(&start);

    char *segmented_img = compute_segments(dev_output.cudaPtr(), image.rows, image.cols, dev_output.step);

    time(&end);
    std::cout << "Segmentation time: " << double(end - start) << std::endl;

    cv::Mat output_img = cv::Mat(image.rows, image.cols, CV_8UC3, segmented_img);
    namedWindow("Segmented", cv::WINDOW_NORMAL);
    imshow("Segmented", output_img);
    cv::waitKey();

    imwrite("segmented.png", output_img);

    return 0;
}
