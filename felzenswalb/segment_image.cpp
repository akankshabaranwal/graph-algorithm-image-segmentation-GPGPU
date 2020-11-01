
#include "segment_image.h"
#include <cmath>

double dissimilarity(unsigned char* image, int width, int row1, int col1, int row2, int col2, int channels)
{
    double dis=0;
    for(int dim=0; dim<channels; dim++)
    {
        dis = dis + pow((image[(row1 * width) + col1] - image[(row2 * width) + col2]),2);
    }
    return sqrt(dis);
}

int num_edge=0;

Edge *create_graph(unsigned char* image, int width, int height, int channels)
{
    int size_graph = width * height * 4;
    Edge edges[size_graph];
    int cur_node,right_node, bottom_node, bottom_right_node;
    double wt;

    for (int i = 0; i<height; i++)
    {
        for(int j=0; j<width; j++)
        {
            cur_node = i*width + j;
            right_node = i*width + j + 1;
            bottom_node = (i+1)*width + j;
            bottom_right_node = (i+1)*width + j + 1;
            if(j< (width -1))
            {
                edges[num_edge].u = cur_node;
                edges[num_edge].v = right_node;
                edges[num_edge].wt = dissimilarity(image, width, i, j, i, j+1, channels);
                num_edge++;
            }
            if(i < (height -1))
            {
                edges[num_edge].u = cur_node;
                edges[num_edge].v = bottom_node;
                edges[num_edge].wt = dissimilarity(image, width, i, j, i+1, j, channels );
                num_edge++;
            }
            if ((j< (width-1)) && (i<(height-1)))
            {
                edges[num_edge].u = cur_node;
                edges[num_edge].v = bottom_right_node;
                edges[num_edge].wt = dissimilarity(image, width, i,j,i+1,j+1, channels);
                num_edge++;
                edges[num_edge].u = bottom_node;
                edges[num_edge].v = right_node;
                edges[num_edge].wt = dissimilarity(image, width, i+1,j,i,j+1, channels);
                num_edge++;
            }
        }
    }
    return edges;
}

Edge* sortEdges(Edge edges[])
{   //insertion sort
    Edge tmp;
    int i,j;
    for(i=0; i<num_edge;i++)
    {
        tmp = edges[i];
        j=i-1;
        while(j>=0 && (edges[j].wt > tmp.wt))
        {
            edges[j+1]=edges[j];
            j=j-1;
        }
        edges[j+1]=tmp;
    }
    return edges;
}

double *segment(unsigned char* image, int k, int min_cmp_size, int width, int height, int channels)
{
    Edge* edges = create_graph(image,width, height, channels);
    edges = sortEdges(edges);

}