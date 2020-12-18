from sys import argv

import imageio
import numpy as np

if __name__ == "__main__":
    shape = imageio.imread(argv[1]).shape
    imageio.imsave(argv[2], np.zeros(shape, dtype='uint8'))
