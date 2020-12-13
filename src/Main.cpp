#include <iostream>
#include <getopt.h>
#include <chrono>

#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>

#include "Options.h"
#include "mst.h"

void printUsage() {
    puts("Usage: ./felz -i [input image path] -o [output image path]");
    puts("Options:");
    puts("\t-i: Path to input file (default: data/beach.png)");
    puts("\t-o: Path to output file (default: segmented.png)");
    puts("\t-c: Use host side kernel launches instead of dynamic parallelism");
    puts("\t-s: Show the generated images");
}

const Options handleParams(int argc, char **argv) {
    Options options = Options();
    for(;;)
    {
        switch(getopt(argc, argv, "ci:o:hs")) // note the colon (:) to indicate that 'b' has a parameter and is not a switch
        {
            case 'c': {
                options.useCPU = true;
                continue;
            }
            case 's': {
                options.show = true;
                continue;
            }
            case 'i': {
                options.inFile = std::string(optarg);
                continue;
            }
            case 'o': {
                options.outFile = std::string(optarg);
                continue;
            }
            case '?':
            case 'h':
            default : {
                printUsage();
                break;
            }

            case -1: break;
        }
        break;
    }
    return options;
}

int main(int argc, char **argv)
{
    const Options options = handleParams(argc, argv);

    cv::Mat image;

    image = imread(options.inFile, cv::IMREAD_COLOR);
    std::cout << "Computing on: " << image.rows << "x" << image.cols << std::endl;

    // Start timer
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Gaussian blur
    cv::cuda::GpuMat dev_image, dev_output;
    dev_image.upload(image);
    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(CV_8UC3, CV_8UC3, cv::Size(5, 5), 1.0);
    filter->apply(dev_image, dev_output);

    // Segmentation
    char *segmented_img = compute_segments(dev_output.cudaPtr(), image.rows, image.cols, dev_output.step);

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "Segmentation time: " << time_span.count() << " s" << std::endl;

    cv::Mat output_img = cv::Mat(image.rows, image.cols, CV_8UC3, segmented_img);
    imwrite(options.outFile, output_img);

    if (options.show) {
        cv::Mat output;
        dev_output.download(output);
        namedWindow("Source Image", cv::WINDOW_NORMAL);
        imshow("Source Image", image);
        namedWindow("After Blur (CUDA)", cv::WINDOW_NORMAL);
        imshow("After Blur (CUDA)", output);
        namedWindow("Segmented", cv::WINDOW_NORMAL);
        imshow("Segmented", output_img);
        cv::waitKey();
    }

    return 0;
}
