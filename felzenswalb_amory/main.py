from imageio import imread
from scipy.ndimage import gaussian_filter

def visualise(orig, compare):
    from scipy import misc
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side


    ax1.imshow(orig)
    ax2.imshow(compare)
    plt.show()

def get_rgb(image, n_rows, n_cols):
    r = []
    g = []
    b = []

    for i in range(0, n_rows):
        row_r = []
        row_g = []
        row_b = []

        for j in range(0, n_cols):
            row_r.append(rgb_matrix[i][j][0])
            row_g.append(rgb_matrix[i][j][1])
            row_b.append(rgb_matrix[i][j][2])
        r.append(row_r)
        g.append(row_g)
        b.append(row_b)

    return (r, g, b)

if __name__ == '__main__':
    img_path = 'beach.gif'
    sigma = 0.8

    orig_rgb_matrix = imread('beach.gif')

    # Apply gaussian kernel for smoothing
    rgb_matrix = gaussian_filter(orig_rgb_matrix, sigma=sigma)
    n_rows = rgb_matrix.shape[0]
    n_cols = rgb_matrix.shape[1]

    r, g, b = get_rgb(rgb_matrix, n_rows, n_cols)
    visualise(orig_rgb_matrix, rgb_matrix)


