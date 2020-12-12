//
// Created by gyorgy on 10/12/2020.
//

#ifndef FELZENSWALB_CUSTOM_TYPES_H
#define FELZENSWALB_CUSTOM_TYPES_H

#define CHANNEL_SIZE 3
#define K 250
#define NUM_NEIGHBOURS 8

typedef struct {
    uint weight;
    uint src_id;
    uint src_comp;
    uint dest_id;
    uint dest_comp;
    uint new_int_diff;
} min_edge;

typedef struct{
    min_edge edge;
    uint locked;
} min_edge_wrapper;

#endif //FELZENSWALB_CUSTOM_TYPES_H
