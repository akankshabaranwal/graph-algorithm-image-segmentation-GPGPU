#ifndef SEGMENT_IMAGE_H
#define SEGMENT_IMAGE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

struct Edge{
public:
    int u;
    int v;
    double wt;
};

double dissimilarity(Mat image, int row1, int col1, int row2, int col2);
void create_graph(Edge edges[], Mat InputImage);

struct subset
{   int parent;
    int rank;
    double thresh;
    int sz;
};

// find and union code from https://www.geeksforgeeks.org/boruvkas-algorithm-greedy-algo-9/
int find(struct subset subsets[], int i);
void Union(struct subset subsets[], int x, int y, int wt, double k);

//Actual segmentation code
void SegmentImage(Mat image, double k, int min_cmp_size);

#endif