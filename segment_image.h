#ifndef SEGMENT_IMAGE_H
#define SEGMENT_IMAGE_H

class Edge{
public:
    int u;
    int v;
    double wt;
};

struct subset
{   int parent;
    int rank;
    int MaxWt;
    int numElem;
};

// find and union code from https://www.geeksforgeeks.org/boruvkas-algorithm-greedy-algo-9/
int find(struct subset subsets[], int i);
void Union(struct subset subsets[], int x, int y);

Edge *create_graph(unsigned char* image, int width, int height, int channels);
double dissimilarity(unsigned char* image, int row1, int col1, int row2, int col2, int channels);

#endif