import argparse
import io
import sys

import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def find_segments(file):
    color_image = imageio.imread(file.read())
    W, H, C = color_image.shape

    colors = np.unique(color_image.reshape(-1, 3), axis=0)
    
    segments = np.empty((W, H), dtype=int)
    segments[:] = -1

    for idx, color in enumerate(colors):
        segments[(color_image == color).all(2)] = idx

    assert (segments != -1).all()

    return segments

def mask(segments, segid):
    return segments == segid

def interesect(segments1, segid1, segments2, segid2):
    return mask(segments1, segid1) & mask(segments2, segid2)

# Achievable Segmentation Accuracy
def asa_score(segments_in, segments_gt):
    N_in = np.max(segments_in) + 1
    N_gt = np.max(segments_gt) + 1
    
    numerator = 0
    for k in range(N_in):
        inter_max = 0
        for i in range(N_gt):
            inter_max = max(inter_max, interesect(segments_in, k, segments_gt, i).sum())
            
        numerator += inter_max
        
    return numerator / segments_in.size

# Under Segmentation Error
def underseg_error(segments_in, segments_gt):
    N_in = np.max(segments_in) + 1
    N_gt = np.max(segments_gt) + 1
    
    numerator = 0

    for i in range(N_gt):
        for k in range(N_in):
            intersection = interesect(segments_in, k, segments_gt, i).sum()
            difference = (mask(segments_in, k) & ~mask(segments_gt, i)).sum()
            
            numerator += min(intersection, difference)
            
    return numerator / segments_in.size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare segmented image with ground truth.')
    parser.add_argument('-H', '--header', required=False, action='store_true')
    parser.add_argument('-i', '--input', required=False, type=argparse.FileType('rb'))
    parser.add_argument('-g', '--ground-truth', required=False, type=argparse.FileType('rb'))
    
    args = parser.parse_args()

    if args.header:
        print('input,ground_truth,asa_score,underseg_error')
        sys.exit(0)
    
    segments_in = find_segments(args.input)
    segments_gt = find_segments(args.ground_truth)

    asa, useg = asa_score(segments_in, segments_gt), underseg_error(segments_in, segments_gt)
    print(f'{args.input.name},{args.ground_truth.name},{asa},{useg}')
