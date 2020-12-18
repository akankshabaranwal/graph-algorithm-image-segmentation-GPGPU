import argparse
import pathlib

import imageio
import numpy as np
import scipy.io
from tqdm import tqdm


def color_image(segments: np.ndarray):
    assert len(segments.shape) == 2
    colors = np.random.randint(0, 256, size=(np.max(segments) + 1, 3))
    return np.array([[colors[val] for val in row] for row in segments], dtype='uint8')

def convert(source: pathlib.Path, dest: pathlib.Path):
    pbar = tqdm(list(source.glob('*.mat')), ascii=True)

    for matfile in pbar:
        mat = scipy.io.matlab.loadmat(matfile)
        mat = mat['groundTruth'][0]

        for i, entry in enumerate(mat):
            segments, _ = entry[0][0]
            image = color_image(segments)
            outfile = dest / f'{matfile.stem}_{i}.png'
            pbar.set_description(str(outfile))
            imageio.imsave(outfile, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert .mat files from BSDS500 into colored segmentations.')
    parser.add_argument('-s', '--source', required=True, help='Source folder containing the BSDS .mat files.')
    parser.add_argument('-d', '--dest', required=True, help='Destination folder for the generated images.')
    args = parser.parse_args()
    
    source = pathlib.Path(args.source)
    dest = pathlib.Path(args.dest)

    convert(source, dest)
