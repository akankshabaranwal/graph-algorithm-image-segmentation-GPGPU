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
#include <unistd.h>
#include "segment-image.h"

float sigma;
float k;
int min_size;
int warmup;
int benchmark;
bool partial;
char* in_path;
char* out_path;

void printUsage() {
    puts("Usage: ./segment -i [input image path] -o [output image path]");
    puts("Options:");
    puts("\t-i: Path to input file (eg: data/beach.ppm)");
    puts("\t-o: Path to output file (eg: segmented.ppm)");
    puts("\t-s: Sigma");
    puts("\t-k: K");
    puts("\t-m: Min segment size");
    puts("Benchmarking options");
    puts("\t-w: Number of iterations to perform during warmup");
    puts("\t-b: Number of iterations to perform during benchmarking");
    puts("\t-p: If want to do partial timings");
    exit(1);
}

void handleParams(int argc, char **argv) {
    for(;;)
    {
        switch(getopt(argc, argv, "phi:o:s:k:m:w:b:"))
        {
            case 'i': {
                in_path = optarg;
                continue;
            }
            case 'o': {
                out_path = optarg;
                continue;
            }
            case 's': {
                sigma = atof(optarg);
                continue;
            }
            case 'k': {
                k = atof(optarg);
                continue;
            }
            case 'm': {
                min_size = atoi(optarg);
                continue;
            }
            case 'w': {
                warmup = atoi(optarg);
                continue;
            }
            case 'b': {
                benchmark = atoi(optarg);
                continue;
            }
            case 'p': {
                partial = true;
                continue;
            }
            case '?':
            case 'h':
            default : {
                printUsage();
                break;
            }

            case -1:  {
                break;
            }
        }
        break;
    }
}


int main(int argc, char **argv) {
  handleParams(argc, argv);

  fprintf(stderr, "loading input image.\n");
  image<rgb> *input = loadPPM(in_path);
	
  fprintf(stderr, "processing\n");
  int num_ccs; 

  for (int i = 0; i < warmup; i++) {
    image<rgb> *w = segment_image(input, sigma, k, min_size, &num_ccs, 0, true);
  }

  if (partial) {
    printf("gaussian, graph, segmentation, output\n");  
  } else {
    printf("total\n");
  }

  for (int i = 0; i < benchmark; i++) {
    if (partial) {
      image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs, partial, false);
      printf("\n");
      if (i == benchmark-1) {
        savePPM(seg, out_path);
      }
    } else {
      std::chrono::high_resolution_clock::time_point start, end;
      start = std::chrono::high_resolution_clock::now();
      image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs, partial, false);
      end = std::chrono::high_resolution_clock::now();
      int time_span = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      printf("%d\n", time_span);
      if (i == benchmark-1) {
        savePPM(seg, out_path);
      }
    }
  }

  fprintf(stderr,"got %d components\n", num_ccs);
  fprintf(stderr,"done! uff...thats hard work.\n");

  return 0;
}

