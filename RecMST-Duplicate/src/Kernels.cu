/***********************************************************************************
  Implementing Minimum Spanning Tree on CUDA using primitive operations for the 
  algorithm given in "Fast Minimum Spanning Tree Computation", by Pawan Harish, 
  P.J. Narayanan, Vibhav Vineet, and Suryakant Patidar.

  Chapter 7 of Nvidia GPU Computing Gems, Jade Edition, 2011.
  
  Copyright (c) 2011 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND, EXPRESSED, IMPLIED OR 
  OTHERWISE.

  Kernels for MST implementation, by Pawan Harish.
 ************************************************************************************/

#ifndef _KERNELS_H_
#define _KERNELS_H_

#define MOVEBITS 26 						// Amount of bits in X for vertex ID
#define NO_OF_BITS_TO_SPLIT_ON 32			// Amount of bits for L split (32 bits one vertex, 32 other)
#define NO_OF_BITS_MOVED_FOR_VERTEX_IDS 26
#define MAX_THREADS_PER_BLOCK 1024 			// IMPORTANT TO SET CORRECTLY
#define INF 10000000						// Make sure larger than amount of edges, maybe best equal to max possible vertex ID // TODO: set to 2^MOVEBITS - 1
#define CHANNEL_SIZE 3						// Amount of color channels, 3 for RGB
#define SCALE 8 							// Make sure to set scale so weight less than assigned amount of bits

////////////////////////////////////////////////////////////////////////////////////////////
// Graph creation kernels
////////////////////////////////////////////////////////////////////////////////////////////
__global__ void createCornerGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, unsigned int *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if (tid < 4) {
		unsigned int row = 0;
		unsigned int col = 0;
		unsigned int write_offset = 0;

		if (tid == 1) {
			col = no_of_cols - 1;
			write_offset = 3 * (no_of_cols - 1) - 1;
		}
		if (tid == 2) {
			row = no_of_rows - 1;
			write_offset = 4 + 6 * (no_of_rows-2) + 3 * (no_of_cols-2) + 4 * (no_of_rows-2) * (no_of_cols-2);
		}
		if (tid == 3) {
			col = no_of_cols - 1;
			row = no_of_rows - 1;
			write_offset = 6 + 6 * (no_of_rows-2) + 6 * (no_of_cols-2) + 4 * (no_of_rows-2) * (no_of_cols-2);
		}

		unsigned int left_node = row * no_of_cols + col - 1;
        unsigned int right_node = row * no_of_cols + col + 1;
        unsigned int top_node = (row - 1) * no_of_cols + col;
        unsigned int bottom_node = (row+1) * no_of_cols + col;

      	unsigned int this_img_idx = row * pitch + col * CHANNEL_SIZE;
    	unsigned char this_r = image[this_img_idx];
    	unsigned char this_g = image[this_img_idx + 1];
    	unsigned char this_b = image[this_img_idx + 2];

    	unsigned char other_r;
    	unsigned char other_g;
    	unsigned char other_b;
    	unsigned int other_img_idx;
    	double distance;

    	unsigned long cur_vertex_idx = row * no_of_cols + col;
        d_vertex[cur_vertex_idx] = write_offset;

    	// Left node
    	if (tid == 1 || tid == 3) {
    		d_edge[write_offset] = left_node;

	    	other_img_idx = row * pitch + (col - 1) * CHANNEL_SIZE;
	        other_r = image[other_img_idx];
	        other_g = image[other_img_idx + 1];
	        other_b = image[other_img_idx + 2];
	    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
	    	d_weight[write_offset] = (unsigned int) round(distance);
    	}
    	
    	// Right node
    	if (tid == 0 || tid == 2) {
    		d_edge[write_offset+1] = right_node;

	        other_img_idx = row * pitch + (col + 1) * CHANNEL_SIZE;
	        other_r = image[other_img_idx];
	        other_g = image[other_img_idx + 1];
	        other_b = image[other_img_idx + 2];
	    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
	    	d_weight[write_offset+1] = (unsigned int) round(distance);
    	}
       

    	// Top node
    	if (tid == 2 || tid == 3) {
    		d_edge[write_offset+2] = top_node;

	        other_img_idx = (row-1) * pitch + col * CHANNEL_SIZE;
	        other_r = image[other_img_idx];
	        other_g = image[other_img_idx + 1];
	        other_b = image[other_img_idx + 2];
	    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
	    	d_weight[write_offset+2] = (unsigned int) round(distance);
    	}
        

    	// Bottom node
    	if (tid == 0 || tid == 1) {
    		d_edge[write_offset+3] = bottom_node;

	        other_img_idx = (row+1) * pitch + col * CHANNEL_SIZE;
	        other_r = image[other_img_idx];
	        other_g = image[other_img_idx + 1];
	        other_b = image[other_img_idx + 2];
	    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
	    	d_weight[write_offset+3] = (unsigned int) round(distance);
    	}
	}
}


__global__ void createFirstRowGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, unsigned int *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch) 
{
	unsigned int row = 0;
	unsigned int col = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;

    if (col > 0 && col < no_of_cols) {
    	unsigned int left_node = row * no_of_cols + col - 1;
        unsigned int right_node = row * no_of_cols + col + 1;
        unsigned int bottom_node = (row+1) * no_of_cols + col;

      	unsigned int this_img_idx = row * pitch + col * CHANNEL_SIZE;
    	unsigned char this_r = image[this_img_idx];
    	unsigned char this_g = image[this_img_idx + 1];
    	unsigned char this_b = image[this_img_idx + 2];

    	unsigned int write_offset = 2 + (col-1) * 3;

    	unsigned char other_r;
    	unsigned char other_g;
    	unsigned char other_b;
    	unsigned int other_img_idx;
    	double distance;

    	unsigned long cur_vertex_idx = row * no_of_cols + col;
        d_vertex[cur_vertex_idx] = write_offset;

    	// Left node
    	d_edge[write_offset] = left_node;

    	other_img_idx = row * pitch + (col - 1) * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset] = (unsigned int) round(distance);

    	// Right node
        d_edge[write_offset+1] = right_node;

        other_img_idx = row * pitch + (col + 1) * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset+1] = (unsigned int) round(distance);

    	// Bottom node
        d_edge[write_offset+2] = bottom_node;

        other_img_idx = (row+1) * pitch + col * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset+2] = (unsigned int) round(distance);
    }
}

__global__ void createLastRowGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, unsigned int *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch) 
{
	unsigned int row = no_of_rows-1;;
    unsigned int col = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;

    if (col > 0 && col < no_of_cols) {
    	unsigned int left_node = row * no_of_cols + col - 1;
        unsigned int right_node = row * no_of_cols + col + 1;
        unsigned int top_node = (row - 1) * no_of_cols + col;

      	unsigned int this_img_idx = row * pitch + col * CHANNEL_SIZE;
    	unsigned char this_r = image[this_img_idx];
    	unsigned char this_g = image[this_img_idx + 1];
    	unsigned char this_b = image[this_img_idx + 2];

    	unsigned int first_row_offset = 4 + 3 * (no_of_cols-2);
    	unsigned int extra_cur_row_offset = 3 + (row-1) * (6 + 4 * (no_of_cols-2));
    	unsigned int extra_cur_col_offset = 3 * (col-1);
    	unsigned int write_offset = first_row_offset + extra_cur_row_offset + extra_cur_col_offset;

    	unsigned char other_r;
    	unsigned char other_g;
    	unsigned char other_b;
    	unsigned int other_img_idx;
    	double distance;

    	unsigned long cur_vertex_idx = row * no_of_cols + col;
        d_vertex[cur_vertex_idx] = write_offset;

    	// Left node
    	d_edge[write_offset] = left_node;

    	other_img_idx = row * pitch + (col - 1) * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset] = (unsigned int) round(distance);

    	// Right node
        d_edge[write_offset+1] = right_node;

        other_img_idx = row * pitch + (col + 1) * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset+1] = (unsigned int) round(distance);

    	// Top node
        d_edge[write_offset+2] = top_node;

        other_img_idx = (row-1) * pitch + col * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset+2] = (unsigned int) round(distance);
    }
}

__global__ void createFirstColumnGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, unsigned int *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch) 
{
	unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = 0;

    if (row > 0 && row < no_of_rows) {
        unsigned int right_node = row * no_of_cols + col + 1;
        unsigned int top_node = (row - 1) * no_of_cols + col;
        unsigned int bottom_node = (row+1) * no_of_cols + col;

      	unsigned int this_img_idx = row * pitch + col * CHANNEL_SIZE;
    	unsigned char this_r = image[this_img_idx];
    	unsigned char this_g = image[this_img_idx + 1];
    	unsigned char this_b = image[this_img_idx + 2];

    	unsigned int first_row_offset = 4 + 3 * (no_of_cols-2);
    	unsigned int extra_cur_row_offset = (row-1) * (6 + 4 * (no_of_cols-2));
    	unsigned int write_offset = first_row_offset + extra_cur_row_offset;

    	unsigned char other_r;
    	unsigned char other_g;
    	unsigned char other_b;
    	unsigned int other_img_idx;
    	double distance;

    	unsigned long cur_vertex_idx = row * no_of_cols + col;
        d_vertex[cur_vertex_idx] = write_offset;

    	// Right node
        d_edge[write_offset] = right_node;

        other_img_idx = row * pitch + (col + 1) * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset] = (unsigned int) round(distance);

    	// Top node
        d_edge[write_offset+1] = top_node;

        other_img_idx = (row-1) * pitch + col * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset+1] = (unsigned int) round(distance);

    	// Bottom node
        d_edge[write_offset+2] = bottom_node;

        other_img_idx = (row+1) * pitch + col * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset+2] = (unsigned int) round(distance);
    }
}

__global__ void createLastColumnGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, unsigned int *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch) 
{
	unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = no_of_cols - 1;

    if (row > 0 && col > 0 && row < no_of_rows && col < no_of_cols) {
    	unsigned int left_node = row * no_of_cols + col - 1;
        unsigned int top_node = (row - 1) * no_of_cols + col;
        unsigned int bottom_node = (row+1) * no_of_cols + col;

      	unsigned int this_img_idx = row * pitch + col * CHANNEL_SIZE;
    	unsigned char this_r = image[this_img_idx];
    	unsigned char this_g = image[this_img_idx + 1];
    	unsigned char this_b = image[this_img_idx + 2];

    	unsigned int first_row_offset = 4 + 3 * (no_of_cols-2);
    	unsigned int extra_cur_row_offset = 3 + (row-1) * (6 + 4 * (no_of_cols-2));
    	unsigned int extra_cur_col_offset = 4 * (col-1);
    	unsigned int write_offset = first_row_offset + extra_cur_row_offset + extra_cur_col_offset;

    	unsigned char other_r;
    	unsigned char other_g;
    	unsigned char other_b;
    	unsigned int other_img_idx;
    	double distance;

    	unsigned long cur_vertex_idx = row * no_of_cols + col;
        d_vertex[cur_vertex_idx] = write_offset;

    	// Left node
    	d_edge[write_offset] = left_node;

    	other_img_idx = row * pitch + (col - 1) * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset] = (unsigned int) round(distance);

    	// Top node
        d_edge[write_offset+1] = top_node;

        other_img_idx = (row-1) * pitch + col * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset+1] = (unsigned int) round(distance);

    	// Bottom node
        d_edge[write_offset+2] = bottom_node;

        other_img_idx = (row+1) * pitch + col * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset+2] = (unsigned int) round(distance);
    }
}

__global__ void createInnerGraphKernel(unsigned char *image, unsigned int *d_vertex, unsigned int *d_edge, unsigned int *d_weight, unsigned int no_of_rows, unsigned int no_of_cols, size_t pitch) 
{
	unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row > 0 && col > 0 && row < no_of_rows && col < no_of_cols) {
    	unsigned int left_node = row * no_of_cols + col - 1;
        unsigned int right_node = row * no_of_cols + col + 1;
        unsigned int top_node = (row - 1) * no_of_cols + col;
        unsigned int bottom_node = (row+1) * no_of_cols + col;

      	unsigned int this_img_idx = row * pitch + col * CHANNEL_SIZE;
    	unsigned char this_r = image[this_img_idx];
    	unsigned char this_g = image[this_img_idx + 1];
    	unsigned char this_b = image[this_img_idx + 2];

    	unsigned int first_row_offset = 4 + 3 * (no_of_cols-2);
    	unsigned int extra_cur_row_offset = 3 + (row-1) * (6 + 4 * (no_of_cols-2));
    	unsigned int extra_cur_col_offset = 4 * (col-1);
    	unsigned int write_offset = first_row_offset + extra_cur_row_offset + extra_cur_col_offset;

    	unsigned char other_r;
    	unsigned char other_g;
    	unsigned char other_b;
    	unsigned int other_img_idx;
    	double distance;

    	unsigned long cur_vertex_idx = row * no_of_cols + col;
        d_vertex[cur_vertex_idx] = write_offset;

    	// Left node
    	d_edge[write_offset] = left_node;

    	other_img_idx = row * pitch + (col - 1) * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset] = (unsigned int) round(distance);

    	// Right node
        d_edge[write_offset+1] = right_node;

        other_img_idx = row * pitch + (col + 1) * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset+1] = (unsigned int) round(distance);

    	// Top node
        d_edge[write_offset+2] = top_node;

        other_img_idx = (row-1) * pitch + col * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset+2] = (unsigned int) round(distance);

    	// Bottom node
        d_edge[write_offset+3] = bottom_node;

        other_img_idx = (row+1) * pitch + col * CHANNEL_SIZE;
        other_r = image[other_img_idx];
        other_g = image[other_img_idx + 1];
        other_b = image[other_img_idx + 2];
    	distance = SCALE * sqrt(pow((this_r - other_r), 2) + pow((this_g - other_g), 2) + pow((this_b - other_b), 2));
    	d_weight[write_offset+3] = (unsigned int) round(distance);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
// Segment extraction kernels
////////////////////////////////////////////////////////////////////////////////////////////
__global__ void RandFloatToRandRGB(char* d_component_colours, float *d_component_colours_float, unsigned int n_numbers) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if (tid < n_numbers) {
		float color = 255 *d_component_colours_float[tid];
		d_component_colours[tid] = (char) color;
	}
}

__global__ void InitPrevLevelComponents(unsigned int* d_prev_level_component, unsigned int no_of_rows, unsigned int no_of_cols) 
{
	unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < no_of_rows && col < no_of_cols) {
		d_prev_level_component[row * no_of_cols + col] = row * no_of_cols + col;
	}
}

__global__ void CreateLevelOutput(char *d_output_image, char *d_component_colours, unsigned int* d_level, unsigned int* d_prev_level_component, unsigned int no_of_rows, unsigned int no_of_cols) 
{
	unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < no_of_rows && col < no_of_cols) {

    	unsigned int prev_component = d_prev_level_component[row * no_of_cols + col];
		unsigned int new_component = d_level[prev_component];

		int img_pos = CHANNEL_SIZE * (row * no_of_cols + col);
		int colour_pos = CHANNEL_SIZE * new_component;

		d_output_image[img_pos] = d_component_colours[colour_pos];
		d_output_image[img_pos + 1] = d_component_colours[colour_pos+1];
		d_output_image[img_pos + 2] = d_component_colours[colour_pos+2];

        d_prev_level_component[row * no_of_cols + col] = new_component;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// Append the Weight And Vertex ID into segmented min scan input array, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////////////////
__global__ void AppendKernel_1(unsigned long long int *d_segmented_min_scan_input, unsigned int *d_weight, unsigned int *d_edges, unsigned int no_of_edges) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges) {
		unsigned long long int val=d_weight[tid];

        val = val<<MOVEBITS; // TODO
        val = val|tid; // TODO

		val=val<<MOVEBITS;
		val=val|d_edges[tid];
     
		d_segmented_min_scan_input[tid]=val;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Make the flag for Input to the segmented min scan, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////
__global__ void ClearArray(unsigned int *d_array, unsigned int size) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<size) {
		d_array[tid]=0;
	}
}

__global__ void ClearEdgeStuff(unsigned int *d_edge, unsigned int *d_weight, unsigned int *d_edge_mapping_copy, unsigned int *d_pick_array, unsigned int size) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<size) {
		d_edge[tid]=0;
		d_weight[tid]=0;
		d_edge_mapping_copy[tid]=0;
		d_pick_array[tid]=0;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Make the flag for Input to the segmented min scan, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeFlag_3(unsigned int *d_edge_flag, unsigned int *d_vertex, unsigned int no_of_vertices) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		unsigned int pointingvertex = d_vertex[tid];
		d_edge_flag[pointingvertex]=1;
	}
}


////////////////////////////////////////////////////////////////////////////////
// Make the Successor array, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeSucessorArray(unsigned int *d_successor, unsigned int *d_vertex, unsigned long long int *d_segmented_min_scan_output, unsigned int no_of_vertices, unsigned int no_of_edges) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		unsigned int end; // Result values always stored at end of each segment
		if(tid<no_of_vertices-1) {
			end = d_vertex[tid+1]-1; // Get end of my segment
		} else {
			end = no_of_edges-1; // Last segment: end = last edge
		}
		unsigned long long int mask = pow(2.0,MOVEBITS)-1; // Mask to extract vertex ID MWOE // TODO
		d_successor[tid] = d_segmented_min_scan_output[end]&mask; // Get vertex part of each (weight|to_vertex_id) element
	}
}

////////////////////////////////////////////////////////////////////////////////
// Remove Cycles Using Successor array, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void RemoveCycles(unsigned int *d_successor, unsigned int no_of_vertices) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x; // TID: vertex ID
	if(tid<no_of_vertices) {
		unsigned int succ = d_successor[tid];
		unsigned int nextsucc = d_successor[succ];
		if(tid == nextsucc) { //Found a Cycle
			//Give the minimum one its own value, breaking the cycle and setting the Representative Vertices
			if(tid < succ) {
				d_successor[tid]=tid;
			} else {
				d_successor[succ]=succ;
			}
		}
	}
}


__global__ void SuccToCopy(unsigned int *d_successor, unsigned int *d_successor_copy, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		d_successor_copy[tid] = d_successor[tid];
	}
}

////////////////////////////////////////////////////////////////////////////////
// Propagate Representative IDs by setting S(u)=S(S(u)), Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void PropagateRepresentativeID(unsigned int *d_successor, unsigned int *d_successor_copy, bool *d_succchange, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		unsigned int succ = d_successor[tid];
		unsigned int newsucc = d_successor[succ];
		if(succ!=newsucc) { //Execution goes on
            printf("d_successor_copy[%d]=%d \n", tid, newsucc);
			d_successor_copy[tid] = newsucc; //cannot have input and output in the same array!!!!!
			*d_succchange=true;
		}
	}
}

__global__ void CopyToSucc(unsigned int *d_successor, unsigned int *d_successor_copy, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
		d_successor[tid] = d_successor_copy[tid];
}


////////////////////////////////////////////////////////////////////////////////
// Append Vertex IDs with SuperVertex IDs, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void AppendVertexIDsForSplit(unsigned long long int *d_vertex_split, unsigned int *d_successor, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		unsigned long long int val;
		val = d_successor[tid]; // representative
		val = val<<NO_OF_BITS_TO_SPLIT_ON;
		val |= tid; // u
		d_vertex_split[tid]=val;
	}
}


////////////////////////////////////////////////////////////////////////////////
// Mark New SupervertexID per vertex, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeSuperVertexIDPerVertex(unsigned int *d_new_supervertexIDs, unsigned long long int *d_vertex_split, unsigned int *d_vertex_flag,unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
	{
		unsigned long long int mask = pow(2.0, NO_OF_BITS_TO_SPLIT_ON)-1;
		unsigned long long int vertexid = d_vertex_split[tid]&mask;
		d_vertex_flag[vertexid] = d_new_supervertexIDs[tid];
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copy New SupervertexID per vertex, resolving read after write inconsistancies, Runs for Vertex Length  // IMPORTANT! RAW
////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void CopySuperVertexIDPerVertex(unsigned int *d_new_supervertexIDs, unsigned int *d_vertex_flag, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		d_new_supervertexIDs[tid] = d_vertex_flag[tid];
	}
}


////////////////////////////////////////////////////////////////////////////////
// Make flag for Scan, assigning new ids to supervertices, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeFlagForScan(unsigned int *d_vertex_flag, unsigned long long int *d_split_input,unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
	{
		if(tid>0)
		{
			//unsigned long long int mask = pow(2.0,NO_OF_BITS_TO_SPLIT_ON)-1;
			unsigned long long int val = d_split_input[tid-1];
			unsigned long long int supervertexid_prev  = val>>NO_OF_BITS_TO_SPLIT_ON;
			val = d_split_input[tid];
			unsigned long long int supervertexid  = val>>NO_OF_BITS_TO_SPLIT_ON;
			if(supervertexid_prev!=supervertexid)
				d_vertex_flag[tid]=1;
		}
	}
}


////////////////////////////////////////////////////////////////////////////////
// Make flag to assign old vertex ids, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeFlagForUIds(unsigned int *d_edge_flag, unsigned int *d_vertex, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices) {
		if(tid>0) {
			unsigned int pointingvertex = d_vertex[tid];
			d_edge_flag[pointingvertex]=1;
		}
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copy Edge Array to somewhere to resolve read after write inconsistancies, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void CopyEdgeArray(unsigned int *d_edge, unsigned int *d_edge_mapping_copy, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges)
		d_edge_mapping_copy[tid] = d_edge[tid];
}

////////////////////////////////////////////////////////////////////////////////
// Remove self edges based on new supervertex ids, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////
__global__ void RemoveSelfEdges(unsigned int *d_edge, unsigned int *d_old_uIDs, unsigned int *d_new_supervertexIDs, unsigned int *d_edge_mapping_copy, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges) {
		unsigned int uid = d_old_uIDs[tid];
		unsigned int vid = d_edge[tid];
		unsigned int usuperid = d_new_supervertexIDs[uid];
		unsigned int vsuperid = d_new_supervertexIDs[vid];
		if(usuperid == vsuperid){
			d_edge_mapping_copy[tid]=INF; //Nullify the edge if both vertices have same supervertex id, do not use the same array for output
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copy Edge Array Back, resolving read after write inconsistancies, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void CopyEdgeArrayBack(unsigned int *d_edge, unsigned int *d_edge_mapping_copy, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges)
		d_edge[tid]=d_edge_mapping_copy[tid];
}


////////////////////////////////////////////////////////////////////////////////
// Append U,V,W for duplicate edge removal, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////
__global__ void AppendForDuplicateEdgeRemoval(unsigned long long int *d_appended_uvw, unsigned int *d_edge, unsigned int *d_old_uIDs, unsigned int *d_weight, unsigned int *d_new_supervertexIDs, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges) {
		unsigned long long int val;
		unsigned int u,v,superuid=INF,supervid=INF;
		u = d_old_uIDs[tid];
		v = d_edge[tid];

		if (v == INF) { // TODO: maybe useful. else (u, INF, w)
			u = INF;
		}

		if(u!=INF && v!=INF) {
			superuid = d_new_supervertexIDs[u];
			supervid = d_new_supervertexIDs[v];
		}
		val = superuid;
		val = val<<NO_OF_BITS_MOVED_FOR_VERTEX_IDS;
		val |= supervid;
		val = val<<(64-(NO_OF_BITS_MOVED_FOR_VERTEX_IDS+NO_OF_BITS_MOVED_FOR_VERTEX_IDS));
		val |= d_weight[tid];
		d_appended_uvw[tid]=val;
	}
}


////////////////////////////////////////////////////////////////////////////////
// Mark the starting edge for each uv combination, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MarkEdgesUV(unsigned int *d_edge_flag, unsigned long long int *d_appended_uvw, unsigned int *d_size, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges) {
		if(tid>0) {
			unsigned long long int test = INF;
			test = test << NO_OF_BITS_MOVED_FOR_VERTEX_IDS;
			test |=INF;
			unsigned long long int test1 = d_appended_uvw[tid]>>(64-(NO_OF_BITS_MOVED_FOR_VERTEX_IDS+NO_OF_BITS_MOVED_FOR_VERTEX_IDS)); // uv[i]
			unsigned long long int test2 = d_appended_uvw[tid-1]>>(64-(NO_OF_BITS_MOVED_FOR_VERTEX_IDS+NO_OF_BITS_MOVED_FOR_VERTEX_IDS)); // uv[i-1]

			if(test1>test2) {
				d_edge_flag[tid]=1;
			}

			if(test1 == test) { // TODO: might be different if change line 334. Not sure if correct now either
				atomicMin(d_size,tid); //also to know the last element in the array, i.e. the size of new edge list
			}
		} else {
			d_edge_flag[tid]=1;
		}
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////
// Compact the edgelist and weight list, keep a mapping for each edge, Runs for d_size Length
///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void CompactEdgeList(unsigned int *d_edge, unsigned int *d_weight, 
								unsigned int *d_old_uIDs, unsigned int *d_edge_flag, unsigned long long int *d_appended_uvw,
								unsigned int *d_pick_array, unsigned int *d_size, 
								unsigned int *d_edge_list_size, unsigned int *d_vertex_list_size)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<*d_size) {
		if(	d_edge_flag[tid]==1) {
			unsigned long long int UVW = d_appended_uvw[tid];
			unsigned int writepos = d_old_uIDs[tid];
			unsigned long long int mask = pow(2.0,64-(NO_OF_BITS_MOVED_FOR_VERTEX_IDS+NO_OF_BITS_MOVED_FOR_VERTEX_IDS))-1;
			unsigned long long int w  = UVW&mask;
			unsigned long long int test = UVW>>(64-(NO_OF_BITS_MOVED_FOR_VERTEX_IDS+NO_OF_BITS_MOVED_FOR_VERTEX_IDS));
			unsigned long long int mask2 = pow(2.0,NO_OF_BITS_MOVED_FOR_VERTEX_IDS)-1;
			unsigned long long int v = test&mask2;
			unsigned long long int u = test>>NO_OF_BITS_MOVED_FOR_VERTEX_IDS;
			if(u!=INF && v!=INF) {
				//Copy the edge_mapping into a temporary array, used to resolve read after write inconsistancies
				d_pick_array[writepos]=u; // reusing this to store u's
				d_edge[writepos] = v;
				d_weight[writepos] = w;
				//max writepos will give the new edge list size
				atomicMax(d_edge_list_size,(writepos+1));
				atomicMax(d_vertex_list_size,(v+1));
				// Orig: atomicMax(d_vertex_list_size,(u+1)); //how can max(v) be > max(u), error!!!!! TODO check this whole thing
			}
		}		
	}
}

////////////////////////////////////////////////////////////////////////////////
//Copy the temporary array to the actual mapping array, Runs for Edge length
////////////////////////////////////////////////////////////////////////////////
__global__ void CopyEdgeMap(unsigned int *d_edge_mapping, unsigned int *d_edge_mapping_copy, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges)
		d_edge_mapping[tid] = d_edge_mapping_copy[tid]; 
}

////////////////////////////////////////////////////////////////////////////////
//Make Flag for Vertex List Compaction, Runs for Edge length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeFlagForVertexList(unsigned int *d_pick_array, unsigned int *d_edge_flag, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges) {
		if(tid>0) {
			if(d_pick_array[tid] != d_pick_array[tid-1]) { //This line may be causing problems TODO: maybe != such as in python code but should be fine. Change back to > for orig
				d_edge_flag[tid]=1;
			}
		} else {
			d_edge_flag[tid]=1;
			//atomicMax(d_edge_list_size,(tid));
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//Vertex List Compaction, Runs for Edge length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeVertexList(unsigned int *d_vertex, unsigned int *d_pick_array, unsigned int *d_edge_flag, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges) {
		if(d_edge_flag[tid]==1) {
			unsigned int writepos=d_pick_array[tid]; //get the u value
			d_vertex[writepos]=tid; //write the index to the u'th value in the array to create the vertex list
			//atomicMax(d_vertex_list_size,(writepos+1));
		}
	}
}


#endif // #ifndef _KERNELS_H_