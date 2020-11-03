//
// Created by Amory Hoste on 25/10/2020.
//

#ifndef FELZENSWALB_GAUSSIAN_FILTER_H
#define FELZENSWALB_GAUSSIAN_FILTER_H

double* gaussian_filter(int w, int h, double sigma);
void convolve2d(double* in, double* out, int inw, int inh, double* filter, int fw, int fh);

#endif //FELZENSWALB_GAUSSIAN_FILTER_H
