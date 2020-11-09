#include "segment_image.h"
#include <cmath>
#include <iostream>

using namespace std;

int num_edge = 0;
int num_vertices = 0;

double dissimilarity(unsigned char *image, int width, int row1, int col1, int row2, int col2, int channels) {
    double dis = 0;
    for (int dim = 0; dim < channels; dim++) {
        dis = dis + pow((image[(row1 * width) + col1] - image[(row2 * width) + col2]), 2);
    }
    return sqrt(dis);
}

// Creating the graph
void create_graph(Edge edges[], unsigned char *image, int width, int height, int channels) {
    //TODO: Update this appropriately. This is the upper bound on the number of edges.
    num_vertices = width * height; //Initially each vertex is a pixel
    int cur_node, right_node, bottom_node, bottom_right_node;
    double wt;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cur_node = i * width + j;
            right_node = i * width + j + 1;
            bottom_node = (i + 1) * width + j;
            bottom_right_node = (i + 1) * width + j + 1;
            if (j < (width - 1)) {
                edges[num_edge].u = cur_node;
                edges[num_edge].v = right_node;
                edges[num_edge].wt = dissimilarity(image, width, i, j, i, j + 1, channels);
                num_edge++;
            }
            if (i < (height - 1)) {
                edges[num_edge].u = cur_node;
                edges[num_edge].v = bottom_node;
                edges[num_edge].wt = dissimilarity(image, width, i, j, i + 1, j, channels);
                num_edge++;
            }
            if ((j < (width - 1)) && (i < (height - 1))) {
                edges[num_edge].u = cur_node;
                edges[num_edge].v = bottom_right_node;
                edges[num_edge].wt = dissimilarity(image, width, i, j, i + 1, j + 1, channels);
                num_edge++;
                edges[num_edge].u = bottom_node;
                edges[num_edge].v = right_node;
                edges[num_edge].wt = dissimilarity(image, width, i + 1, j, i, j + 1, channels);
                num_edge++;
            }
        }
    }
}


// Graph segmentation functions
// A utility function to find set of an element i (uses path compression technique)
int find(struct subset subsets[], int i) {
    // find root and make root as parent of i (path compression)
    if (subsets[i].parent != i)
        subsets[i].parent = find(subsets, subsets[i].parent);

    return subsets[i].parent;
}

// A function that does union of two sets of x and y
// (uses union by rank)
void Union(struct subset subsets[], int x, int y, double wt,
           double k) //TODO: Check if we need to do union by rank or union by size
{
    int xroot = find(subsets, x);
    int yroot = find(subsets, y);

    // Attach smaller rank tree under root of high rank tree (Union by Rank)
    if (subsets[xroot].rank < subsets[yroot].rank) {
        subsets[xroot].parent = yroot;
        subsets[yroot].sz = subsets[xroot].sz + subsets[yroot].sz;
        subsets[yroot].thresh = wt + (k/subsets[yroot].sz);
    } else if (subsets[xroot].rank > subsets[yroot].rank) {
        subsets[yroot].parent = xroot;
        subsets[xroot].sz = subsets[xroot].sz + subsets[yroot].sz;
        subsets[xroot].thresh = wt + (k/subsets[xroot].sz);
    } else// If ranks are same, then make one as root and increment its rank by one
    {
        subsets[yroot].parent = xroot;
        subsets[xroot].rank++;
        subsets[xroot].sz = subsets[xroot].sz + subsets[yroot].sz;
        subsets[xroot].thresh = wt + (k/subsets[xroot].sz);
    }
}

//Image segmentation using Boruvka's
//Generously borrowed from https://www.geeksforgeeks.org/boruvkas-algorithm-greedy-algo-9/
// Pseudo code of segmentation from https://dergipark.org.tr/en/pub/ijamec/issue/25619/271038

void ImageSegment(unsigned char *image, double k, int min_cmp_size, int width, int height, int channels) {
    int size_graph = width * height * 4;
    //TODO: Update this appropriately. This is the upper bound on the number of edges.
    Edge edges[size_graph];
    num_vertices = width * height; //Initially each vertex is a pixel
    create_graph(edges, image, width, height, channels);

    struct subset *subsets = new subset[num_vertices];
    int *cheapest = new int[num_vertices];
    int DidReduce = 1; //This is for checking if the number of components reduced

    // Create V subsets with single elements
    for (int v = 0; v < num_vertices; ++v) {
        subsets[v].parent = v;
        subsets[v].rank = 0;
        subsets[v].thresh = k; // TODO: Check what does this need to be initialized to?
        subsets[v].sz = 1; // At the start each subset has just one element.
        cheapest[v] = -1;
    }
    int numTrees = num_vertices;
    for (int i = 0; i < num_edge; i++) {
        cout << "Printing edge information: " << edges[i].u << ' ' << edges[i].v << ' ' << edges[i].wt << endl;
    }
    while(DidReduce>0){
    //while (numTrees > 1) {
        DidReduce = 0;
        // While condition checking if we can find any more components or no. Everytime initialize cheapest array
        for (int v = 0; v < num_vertices; ++v) {
            cheapest[v] = -1;
        }

        // Traverse through all edges and update cheapest of every component
        for (int i = 0; i < num_edge; i++) {   // Find components (or sets) of two corner of current edge
            int set1 = find(subsets, edges[i].u);
            int set2 = find(subsets, edges[i].v);
            // If two corners of current edge belong to same set, ignore current edge
            if (set1 == set2)
                continue;
                // Else check if current edge is closer to previous cheapest edges of set1 and set2
            else {
                if (cheapest[set1] == -1 || (edges[cheapest[set1]].wt > edges[i].wt))
                    cheapest[set1] = i;
                if (cheapest[set2] == -1 || (edges[cheapest[set2]].wt > edges[i].wt))
                    cheapest[set2] = i;
            }
        }

        //Felzenszwalb segmentation
        for (int i = 0; i < num_vertices; i++) {
            // Check if cheapest for current set exists
            if (cheapest[i] != -1) { // Cheapest[i] selects the smallest adjoining edge
                int set1 = find(subsets, edges[cheapest[i]].u);
                int set2 = find(subsets, edges[cheapest[i]].v);

                if (set1 == set2)
                    continue;
                // Do a union of set1 and set2 and decrease number of trees
                if((edges[cheapest[i]].wt < subsets[set1].thresh) && (edges[cheapest[i]].wt < subsets[set2].thresh)) {
                Union(subsets, set1, set2, edges[cheapest[i]].wt, k);
                numTrees--;
                DidReduce++;
                }
            }
        }
    }

    // based on minsize
    for (int i=0; i<num_edge; i++)
    {
        int set1 = find(subsets, edges[i].u);
        int set2 = find(subsets, edges[i].v);
        if (set1 == set2)
            continue;
        // Do a union of set1 and set2 and decrease number of trees
        if((subsets[set1].sz < min_cmp_size) || ( subsets[set2].sz < min_cmp_size)) {
            Union(subsets, set1, set2, edges[i].wt, k);
        }
    }

    int pixel;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            pixel = (y * width) + x;
            image[pixel] = image[find(subsets,pixel)];
        }
    }
}