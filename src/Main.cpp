#include <iostream>

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

    dev_output.download(output);

    imshow("Source Image", image);
    imshow("After Blur (CUDA)", output);

    waitKey();

    return 0;
}
