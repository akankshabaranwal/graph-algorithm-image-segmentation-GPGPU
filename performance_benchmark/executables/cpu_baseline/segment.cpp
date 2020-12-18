/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#include <cstdio>
#include <cstdlib>
#include <image.h>
#include <misc.h>
#include <pnmfile.h>
#include <chrono>
#include "segment-image.h"

int main(int argc, char **argv) {
  if (argc != 9) {
    fprintf(stderr, "usage: %s sigma k min input(ppm) output(ppm) warmup benchmark partial(0:complete,1:partial)\n", argv[0]);
    return 1;
  }
  
  float sigma = atof(argv[1]);
  float k = atof(argv[2]);
  int min_size = atoi(argv[3]);
  int warmup = atoi(argv[6]);
  int benchmark = atoi(argv[7]);
	int partial = atoi(argv[8]);

  fprintf(stderr, "loading input image.\n");
  image<rgb> *input = loadPPM(argv[4]);
	
  fprintf(stderr, "processing\n");
  int num_ccs; 

  for (int i = 0; i < warmup; i++) {
    image<rgb> *w = segment_image(input, sigma, k, min_size, &num_ccs, 0);
  }

  if (partial) {
    printf("gaussian, graph, segmentation, output\n");  
  } else {
    printf("total\n");
  }

  for (int i = 0; i < benchmark; i++) {
    if (partial) {
      image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs, partial);
      printf("\n");
      if (i == benchmark-1) {
        savePPM(seg, argv[5]);
      }
    } else {
      std::chrono::high_resolution_clock::time_point start, end;
      start = std::chrono::high_resolution_clock::now();
      image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs, partial);
      end = std::chrono::high_resolution_clock::now();
      int time_span = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      printf("%d\n", time_span);
      if (i == benchmark-1) {
        savePPM(seg, argv[5]);
      }
    }
  }

  fprintf(stderr,"got %d components\n", num_ccs);
  fprintf(stderr,"done! uff...thats hard work.\n");

  return 0;
}

