import argparse
import pathlib
import sys

import imageio
import numpy as np
from skimage.segmentation import felzenszwalb


def color_image(segments: np.ndarray):
    assert len(segments.shape) == 2
    colors = np.random.randint(0, 256, size=(np.max(segments) + 1, 3))
    return np.array([[colors[val] for val in row] for row in segments], dtype='uint8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sci-kit image felzenswalb implementation.")
    
    parser.add_argument('-k', '--scale', default=1, type=int, 
        help="Free parameter. Higher means larger clusters.")
    parser.add_argument('-s', '--sigma', default=0.8, type=float, 
        help="Width (standard deviation) of Gaussian kernel used in preprocessing.")
    parser.add_argument('-m', '--min-size', default=20, type=int, 
        help="Minimum component size. Enforced using postprocessing.")
    
    parser.add_argument('input', help="Input image.")
    parser.add_argument('output', help="Output image.")

    args = parser.parse_args()
    
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    assert input_path.is_file()

    image = imageio.imread(input_path)
    segments = color_image(felzenszwalb(image, args.scale, args.sigma, args.min_size))
    imageio.imwrite(output_path, segments)
