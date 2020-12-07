#include <opencv2/ximgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ximgproc;

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;


void sobel1() {
    String dir = "/Users/amoryhoste/Library/Mobile Documents/com~apple~CloudDocs/Documents/School/Unief/Master/2e master/Design of Parallel and High-Performance Computing/project/graph-algorithm-image-segmentation/sobel/";

    // Inputs
    String modelFilename = dir + "model.yml.gz";
    String inFilename = dir + "beach.jpg";
    String outFilename = dir + "beach_edge.jpg";
    String outFilename_nms = dir + "beach_edge_nms.jpg";

    // Load source color image
    Mat image = imread(inFilename, 1);

    if (image.empty()) {
        CV_Error(Error::StsError, String("Cannot read image file: ") + inFilename);
    }

    // First we declare the variables we are going to use
    Mat src, src_gray;
    Mat grad;
    int ddepth = CV_16S;

    // 1. Remove noise by blurring with a Gaussian filter ( kernel size = 3 ). Else sobel output very noisy
    GaussianBlur(image, src, Size(3, 3), 0, 0);

    // 2. Convert the image to grayscale
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    // 3. Apply sobel in x and y direction
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(src_gray, grad_x, ddepth, 1, 0);
    Sobel(src_gray, grad_y, ddepth, 0, 1);

    // 4. converting back to CV_8U, compute final output
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // 5. Write outputs
    imwrite(outFilename, grad);
}

void sobel2() {
    String dir = "/Users/amoryhoste/Library/Mobile Documents/com~apple~CloudDocs/Documents/School/Unief/Master/2e master/Design of Parallel and High-Performance Computing/project/graph-algorithm-image-segmentation/sobel/";

    // Inputs
    String modelFilename = dir + "model.yml.gz";
    String inFilename = dir + "beach.jpg";
    String outFilename = dir + "beach_edge.jpg";

    // Load source color image
    Mat3b image = imread(inFilename);

    if (image.empty()) {
        CV_Error(Error::StsError, String("Cannot read image file: ") + inFilename);
    }

    //Convert to grayscale
    Mat1b gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    //Compute dx and dy derivatives
    Mat1f dx, dy;
    Sobel(gray, dx, CV_32F, 1, 0);
    Sobel(gray, dy, CV_32F, 0, 1);

    //Compute gradient
    Mat1f magn;
    magnitude(dx, dy, magn);

    //Show gradient
    imwrite(outFilename, magn);
}

int main( int argc, char** argv )
{


}