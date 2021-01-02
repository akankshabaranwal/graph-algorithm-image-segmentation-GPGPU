//
// Created by gyorgy on 10/12/2020.
//

#ifndef FELZENSWALB_CUSTOM_TYPES_H
#define FELZENSWALB_CUSTOM_TYPES_H

#define CHANNEL_SIZE 3
#define NUM_NEIGHBOURS 8

typedef struct {
    uint weight;
    uint src_comp;
    uint dest_comp;
    uint new_int_diff;
} min_edge;

typedef struct {
    uint weight;
    uint dest_comp;
} short_edge;

typedef struct{
    uint2 edge;
} min_edge_wrapper;

#endif //FELZENSWALB_CUSTOM_TYPES_H
