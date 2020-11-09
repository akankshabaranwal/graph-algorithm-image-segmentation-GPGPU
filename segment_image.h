#ifndef SEGMENT_IMAGE_H
#define SEGMENT_IMAGE_H

struct Edge{
public:
    int u;
    int v;
    double wt;
};

double dissimilarity(unsigned char* image, int row1, int col1, int row2, int col2, int channels);
Edge *create_graph(unsigned char* image, int width, int height, int channels);

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
void ImageSegment(unsigned char* image, double k, int min_cmp_size, int width, int height, int channels);

#endif