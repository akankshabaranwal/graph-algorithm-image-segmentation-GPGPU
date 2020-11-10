#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "segment_image.h"

using namespace cv;
using namespace std;

// Libraries for reading & writing images, don't change!
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

int main() {

    string image_path = "data/beach.png";
    string blurred_path = "data/beachBlurred.png";
    Mat InputImage, BlurredImage;
    InputImage=imread(image_path, IMREAD_COLOR);
    BlurredImage = InputImage.clone();
    if(InputImage.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    double Sigma = 0.5;
    GaussianBlur(InputImage, BlurredImage,Size(3,3), Sigma, Sigma);
    //imshow("Blurred image", BlurredImage);
    //int Key = waitKey(0); // Wait for a keystroke in the window
    //if(Key == 's')
    //{
     //  imwrite("data/blurred.png", BlurredImage);
    //}

    int width, height, channels;
    double sigma = 0.5;
    int k = 500;
    int min = 50;
    unsigned char *img;

    // Image segmentation code
    SegmentImage(BlurredImage, k, min); //Segmenting the grayscale image

    return 0;
}
