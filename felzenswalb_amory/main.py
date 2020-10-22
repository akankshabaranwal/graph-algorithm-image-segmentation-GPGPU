from imageio import imread
from scipy.ndimage import gaussian_filter
from felzenswalb import felzenswalb

def visualise(orig, compare):
    from scipy import misc
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side


    ax1.imshow(orig)
    ax2.imshow(compare)
    plt.show()


def get_rgb(rgb_matrix):
    r = rgb_matrix[:, :, 0]
    g = rgb_matrix[:, :, 1]
    b = rgb_matrix[:, :, 2]

    return (r, g, b)

# TODO: min component size
if __name__ == '__main__':
    img_path = 'beach.gif'
    sigma = 0.5
    k = 500

    # Read image and remove alpha channel
    orig_rgb_matrix = imread('beach.gif')[:, :, :3]

    # Apply gaussian kernel for smoothing
    rgb_matrix = gaussian_filter(orig_rgb_matrix, sigma=sigma)
    n_rows = rgb_matrix.shape[0]
    n_cols = rgb_matrix.shape[1]
    r, g, b = get_rgb(rgb_matrix)

    #visualise(orig_rgb_matrix, rgb_matrix)
    components = felzenswalb(b, k)
    for i in range(0, n_rows):
        for j in range(0, n_cols):
            r[i][j] = components.find(i*n_cols+j)

    visualise(orig_rgb_matrix, r)


