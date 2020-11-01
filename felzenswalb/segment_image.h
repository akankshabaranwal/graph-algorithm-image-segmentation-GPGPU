#ifndef SEGMENT_IMAGE_H
#define SEGMENT_IMAGE_H

class Edge{
public:
    int u;
    int v;
    double wt;
};

Edge *create_graph(unsigned char* image, int width, int height, int channels);
double dissimilarity(unsigned char* image, int row1, int col1, int row2, int col2, int channels);

#endif