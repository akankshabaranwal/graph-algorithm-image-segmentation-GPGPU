from imageio import imread
from felzenswalb import felzenswalb
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import ndimage


def visualise(orig, compare, grayscale=False):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side

    if grayscale:
        plt.gray()

    ax1.imshow(orig)
    ax2.imshow(compare)
    plt.show()


def remove_alpha(image):
    if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]

    return image

def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(0, 255)
    rgb[1] = random.randint(0, 255)
    rgb[2] = random.randint(0, 255)
    return rgb

if __name__ == '__main__':
    img_path = 'beach.gif'
    sigma = 0.5
    k = 500
    min = 50

    # Read image and remove alpha channel, somehow not accurate enough if work with ints instead of floats!
    image = remove_alpha(imread(img_path)).astype(float)
    n_rows = image.shape[0]
    n_cols = image.shape[1]

    components = felzenswalb(image, sigma, k, min)

    colors = np.zeros(shape=(n_rows * n_cols, 3))
    for i in range(n_rows * n_cols):
        colors[i, :] = random_rgb()

    output = np.zeros(shape=(n_rows, n_cols, 3), dtype=int)
    for i in range(n_rows):
        for j in range(n_cols):
            output[i][j] = colors[components.find(i*n_cols + j),:]

    visualise(image.astype(int), output)
