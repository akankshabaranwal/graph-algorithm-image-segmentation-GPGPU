#include <opencv2/ximgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ximgproc;

int main() {

    String dir = "/Users/amoryhoste/Library/Mobile Documents/com~apple~CloudDocs/Documents/School/Unief/Master/2e master/Design of Parallel and High-Performance Computing/project/graph-algorithm-image-segmentation/structured-forest-edge/";

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

    // Convert source image to [0;1] range
    image.convertTo(image, DataType<float>::type, 1/255.0);

    // Run main algorithm to detect edges
    Mat edges(image.size(), image.type());
    Ptr<StructuredEdgeDetection> pDollar = createStructuredEdgeDetection(modelFilename);
    pDollar->detectEdges(image, edges);

    // Write outputs
    imwrite(outFilename, 255*edges);

    return 0;
}