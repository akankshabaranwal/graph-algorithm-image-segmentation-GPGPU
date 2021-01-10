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
    puts("\t-k: K from Felzenszwalb algorithm (default: 200)");
    puts("\t-m: Min size for post-processing (default: 200)");
    puts("\t-E: sigma for Gaussian filter (default: 1.0)");
    puts("\t-w: Number of iterations to perform during warmup (default: 1)");
    puts("\t-b: Number of iterations to perform during benchmarking (default: 10)");
    puts("\t-c: Use host side kernel launches instead of dynamic parallelism");
    puts("\t-s: Show the generated images");
    puts("\t-p: Show partial times in the following order: gaussian filter, graph creation, segmentation, image creation");
    exit(1);
}

Options handleParams(int argc, char **argv) {
    Options options = Options();
    for(;;)
    {
        switch(getopt(argc, argv, "pchsi:o:w:b:k:E:m:"))
        {
            case 'p': {
                options.partial = true;
                continue;
            }
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
            case 'w': {
                options.warmupIterations = atoi(optarg);
                continue;
            }
            case 'b': {
                options.benchmarkIterations = atoi(optarg);
                continue;
            }
            case 'k' : {
                options.k = atoi(optarg);
                continue;
            }
            case 'E': {
                options.sigma = atof(optarg);
                continue;
            }
            case 'm': {
                options.min_size = atoi(optarg);
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

char *segment_wrapper(cv::Mat image, Options options, bool isBenchmarking) {
    if (!isBenchmarking) std::cout.setstate(std::ios_base::failbit);
    else std::cout.clear();

    char *img;
    cv::Ptr<cv::cuda::Filter> filter;
    cv::cuda::GpuMat dev_image, dev_output;
    std::chrono::high_resolution_clock::time_point start, end;

    // Start timer
    start = std::chrono::high_resolution_clock::now();

    // Gaussian blur
    dev_image.upload(image);
    filter = cv::cuda::createGaussianFilter(CV_8UC3, CV_8UC3, cv::Size(5, 5), options.sigma);
    filter->apply(dev_image, dev_output);
    if (options.partial)  {
        end = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << time_span.count() << ",";
    }

    // Segmentation
    if (!options.partial) {
        img = compute_segments(dev_output.cudaPtr(), image.rows, image.cols, dev_output.step, options.useCPU,options.k, options.min_size);
    }
    else {
        img = compute_segments_partial(dev_output.cudaPtr(), image.rows, image.cols, dev_output.step, options.useCPU,options.k, options.min_size);
    }

    // Stop timer
    end = std::chrono::high_resolution_clock::now();

    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (!options.partial) std::cout << time_span.count() << std::endl;
    return img;
}

int main(int argc, char **argv)
{
    const Options options = handleParams(argc, argv);

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator (cv::cuda::HostMem::AllocType::PAGE_LOCKED));
    cv::Mat image = imread(options.inFile, cv::IMREAD_COLOR);
    //std::cout << image.rows * image.cols << std::endl;

    char *segmented_img;

    // Warm up
    for (int i = 0; i < options.warmupIterations; ++i) {
        segmented_img = segment_wrapper(image, options, false);
        free_img(segmented_img);
    }

    if (options.partial) {
        puts("gaussian,graph,segmentation,output");
    }
    else {
        puts("total");
    }

    // Benchmark
    for (int i = 0; i < options.benchmarkIterations; ++i) {
        segmented_img = segment_wrapper(image, options, true);
        if (i < options.benchmarkIterations - 1) free_img(segmented_img);
    }

    cv::Mat output_img = cv::Mat(image.rows, image.cols, CV_8UC3, segmented_img);
    imwrite(options.outFile, output_img);

    if (options.show) {
        namedWindow("Source Image", cv::WINDOW_NORMAL);
        imshow("Source Image", image);
        namedWindow("Segmented", cv::WINDOW_NORMAL);
        imshow("Segmented", output_img);
        cv::waitKey();
    }

    return 0;
}
