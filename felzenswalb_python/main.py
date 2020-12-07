from imageio import imread
from felzenswalb import felzenswalb
from felzenswalb_edge import felzenswalb_edge
from superpixel import superpixel
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(10)

def visualise(orig, segment, grayscale=False):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side

    if grayscale:
        ax1.imshow(orig, cmap='gray', vmin=0, vmax=255)
    else:
        ax1.imshow(orig)

    ax2.imshow(segment)
    plt.show()


def remove_alpha(image):
    if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]

    return image


def random_rgb():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]


if __name__ == '__main__':
    img_path = 'data/beach.gif'
    edge_path = 'data/beach_edge.jpg'
    sigma = 1.5
    k = 500
    min = 50

    # Read image and remove alpha channel, somehow not accurate enough if work with ints instead of floats!
    image = remove_alpha(imread(img_path)).astype(float)
    n_rows = image.shape[0]
    n_cols = image.shape[1]

    # Read edge and remove alpha channel
    edge = remove_alpha(imread(edge_path)).astype(float)

    # Edge detection must be of same image as orig image
    assert n_rows == edge.shape[0] and  n_cols == edge.shape[1] and len(edge.shape) == 2

    segmentation_hierarchy = superpixel(image, edge, sigma, k, min)

    #segmentation_hierarchy = felzenswalb(image, sigma, k, min)


    colors = np.zeros(shape=(n_rows * n_cols, 3))
    for i in range(n_rows * n_cols):
        colors[i, :] = random_rgb()

    output = np.zeros(shape=(n_rows, n_cols, 3), dtype=int)
    prev_level_component = [i * n_cols + j for i in range(n_rows) for j in range(n_cols)]

    for level in range(len(segmentation_hierarchy)):
        for i in range(n_rows):
            for j in range(n_cols):
                prev_component = prev_level_component[i*n_cols+j]
                new_component = segmentation_hierarchy[level][prev_component]
                output[i][j] = colors[new_component,:]
                prev_level_component[i*n_cols+j] = new_component

        is_grayscale = len(image.shape) < 3
        visualise(image.astype(int), output, grayscale=is_grayscale)

