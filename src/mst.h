//
// Created by gyorgy on 16/11/2020.
//

#ifndef FELZENSWALB_MST_H
#define FELZENSWALB_MST_H

char *compute_segments(void *input, uint x, uint y, size_t pitch, bool use_cpu, uint k, uint min_size);
char *compute_segments_partial(void *input, uint x, uint y, size_t pitch, bool use_cpu, uint k, uint min_size);
void free_img(char *img);

#endif //FELZENSWALB_MST_H
