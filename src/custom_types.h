//
// Created by gyorgy on 10/12/2020.
//

#ifndef FELZENSWALB_CUSTOM_TYPES_H
#define FELZENSWALB_CUSTOM_TYPES_H

#define CHANNEL_SIZE 3
#define K 20
#define NUM_NEIGHBOURS 8

typedef struct {
    uint weight;
    uint src_id;
    uint src_comp;
    uint dest_id;
    uint dest_comp;
    uint new_int_diff;
} min_edge;

#endif //FELZENSWALB_CUSTOM_TYPES_H
