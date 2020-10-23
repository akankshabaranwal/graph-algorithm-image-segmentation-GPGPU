from disjoint_set import DisjointSet
import numpy as np
from scipy.ndimage import gaussian_filter
np.seterr(all='raise')
import math

def felzenswalb(input_image, sigma, k, min_cmp_size):

    # Apply gaussian kernel for smoothing
    smoothed_image = gaussian_filter(input_image, sigma=sigma)

    return segment(smoothed_image, k, min_cmp_size)


def segment(image, k, min_cmp_size):

    edges = create_graph(image)
    edges = sorted(edges, key=lambda el: el[2])
    m = len(edges)

    components = DisjointSet()
    internal_diff = dict()
    cmp_size = dict()

    for q in range(m):
        src, dst, w = edges[q]
        src_cmp = components.find(src)
        dst_cmp = components.find(dst)

        if src_cmp != dst_cmp:
            src_int_diff = internal_diff.get(src_cmp, 0)
            src_cmp_size = cmp_size.get(src_cmp, 1)
            src_diff = src_int_diff + (k / src_cmp_size)

            dst_int_diff = internal_diff.get(dst_cmp, 0)
            dst_cmp_size = cmp_size.get(dst_cmp, 1)
            dst_diff = dst_int_diff + (k / dst_cmp_size)

            if w <= min(src_diff, dst_diff):
                components.union(src_cmp, dst_cmp)
                merged_cmp = components.find(src)
                internal_diff[merged_cmp] = w
                cmp_size[merged_cmp] = src_cmp_size + dst_cmp_size

                rmv_cmp = src_cmp if merged_cmp == dst_cmp else dst_cmp
                if rmv_cmp in internal_diff:
                    del internal_diff[rmv_cmp]
                    del cmp_size[rmv_cmp]

    # Join small components
    for q in range(m):
        src, dst, w = edges[q]
        src_cmp = components.find(src)
        dst_cmp = components.find(dst)

        if src_cmp != dst_cmp:
            src_cmp_size = cmp_size.get(src_cmp, 1)
            dst_cmp_size = cmp_size.get(dst_cmp, 1)
            if src_cmp_size < min_cmp_size or dst_cmp_size < min_cmp_size:
                components.union(src_cmp, dst_cmp)
                merged_cmp = components.find(src)
                cmp_size[merged_cmp] = src_cmp_size + dst_cmp_size

    return components


def dissimilarity(image, row1, col1, row2, col2):
    is_grayscale = len(image.shape) == 2

    if is_grayscale:
        return abs(image[row1][col1] - image[row2][col2])
    else:
        return math.sqrt(
            np.sum(
                pow(image[row1][col1] - image[row2][col2], 2)
            )
        )


def create_graph(image):
    n_rows = image.shape[0]
    n_cols = image.shape[1]

    edges = []

    for i in range(n_rows):
        for j in range(n_cols):
            cur_node = i * n_cols + j
            right_node = i * n_cols + j + 1
            bottom_node = (i+1) * n_cols + j
            bottom_right_node = (i+1) * n_cols + j + 1

            if j < n_cols - 1:
                edges.append((cur_node, right_node, dissimilarity(image, i, j, i, j+1)))

            if i < n_rows - 1:
                edges.append((cur_node, bottom_node, dissimilarity(image, i, j, i+1, j)))

            if j < n_cols - 1 and i < n_rows - 1:
                edges.append((cur_node, bottom_right_node, dissimilarity(image, i, j, i+1, j+1)))
                edges.append((bottom_node, right_node, dissimilarity(image, i+1, j, i, j+1)))

    return edges