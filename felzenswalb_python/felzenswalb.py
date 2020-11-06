from disjoint_set import DisjointSet
import numpy as np
from scipy.ndimage import gaussian_filter
np.seterr(all='raise')
import math
import time

MAX_DIFF = math.sqrt(3 * pow(255,2))

def felzenswalb(input_image, sigma, k, min_cmp_size):
    # Apply gaussian kernel for smoothing
    smoothed_image = gaussian_filter(input_image, sigma=sigma)

    start_time = time.time()
    # Make sure operations on input image are floating point!
    segmentation = segment_boruvka_2(smoothed_image, k, min_cmp_size)
    print("--- %s seconds ---" % (time.time() - start_time))
    return segmentation


# Options
# 1 get smallest edge in parallel, then decide whether to join that edge, use same component size for all joins
# 2 get smallest edge in parallel, then decide whether to join that edge, update component sizes for joining
# 1 & 2 resemble original algorithm more and faster. 1 seems more similar to parallel version but seriously undersegments
# idea: maybe use some kind of order from smallest to largest edges when joining, but of course bad for parallelism
# 3 get smallest edge that can be joined, seems to undersegment compared to original algorithm. Also much slower
# Note: output will inherently not resemble original algorithm completely because the threshold function that
# uses the component size to decide whether to join two components
def segment_boruvka_1(image, k, min_cmp_size):
    edges = create_graph(image)
    m = len(edges)

    n_rows = image.shape[0]
    n_cols = image.shape[1]

    components = DisjointSet()
    prev_n_components = n_rows * n_cols + 1
    n_components = n_rows * n_cols

    internal_diff = dict()
    cmp_size = dict()

    # How fast converges, still . log . like boruvka?
    # I think at least as fast, because threshold makes it harder and harder as component size grows ...
    while n_components != prev_n_components:
        prev_n_components = n_components

        # Find cheapest edge out of each component
        cheapest_edge = dict() # array would of course better
        for q in range(m):
            src, dst, w = edges[q]
            src_cmp = components.find(src)
            dst_cmp = components.find(dst)

            if src_cmp != dst_cmp:
                if w < cheapest_edge.get(src_cmp, (None, None, MAX_DIFF))[2]:
                    cheapest_edge[src_cmp] = (src_cmp, dst_cmp, w)

                if w < cheapest_edge.get(dst_cmp, (None, None, MAX_DIFF))[2]:
                    cheapest_edge[dst_cmp] = (src_cmp, dst_cmp, w)

        new_cmp_size = cmp_size.copy()
        # Join edges that satisfy Felzenswalb join predicate
        for comp in cheapest_edge:
            src_cmp, dst_cmp, w = cheapest_edge[comp]

            # Needed because components might have been joined
            src_cmp = components.find(src_cmp)
            dst_cmp = components.find(dst_cmp)

            if src_cmp != dst_cmp:
                src_int_diff = internal_diff.get(src_cmp, 0)
                src_cmp_size = cmp_size.get(src_cmp, 1)
                src_diff = src_int_diff + (k / src_cmp_size)

                dst_int_diff = internal_diff.get(dst_cmp, 0)
                dst_cmp_size = cmp_size.get(dst_cmp, 1)
                dst_diff = dst_int_diff + (k / dst_cmp_size)

                if w <= min(src_diff, dst_diff):
                    components.union(src_cmp, dst_cmp)
                    n_components -= 1

                    internal_diff[src_cmp] = max(src_int_diff, w)
                    internal_diff[dst_cmp] = max(dst_int_diff, w)

                    new_cmp_size[src_cmp] = src_cmp_size + dst_cmp_size
                    new_cmp_size[dst_cmp] = src_cmp_size + dst_cmp_size
        cmp_size = new_cmp_size.copy()

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

def segment_boruvka_2(image, k, min_cmp_size):
    edges = create_graph(image)
    m = len(edges)

    n_rows = image.shape[0]
    n_cols = image.shape[1]

    components = DisjointSet()
    prev_n_components = n_rows * n_cols + 1
    n_components = n_rows * n_cols

    internal_diff = dict()
    cmp_size = dict()

    # How fast converges, still . log . like boruvka?
    # I think at least as fast, because threshold makes it harder and harder as component size grows ...
    while n_components != prev_n_components:
        prev_n_components = n_components

        # Find cheapest edge out of each component
        cheapest_edge = dict() # array would of course better
        for q in range(m):
            src, dst, w = edges[q]
            src_cmp = components.find(src)
            dst_cmp = components.find(dst)

            if src_cmp != dst_cmp:
                if w < cheapest_edge.get(src_cmp, (None, None, MAX_DIFF))[2]:
                    cheapest_edge[src_cmp] = (src_cmp, dst_cmp, w)

                if w < cheapest_edge.get(dst_cmp, (None, None, MAX_DIFF))[2]:
                    cheapest_edge[dst_cmp] = (src_cmp, dst_cmp, w)

        # Join edges that satisfy Felzenswalb join predicate
        for comp in cheapest_edge:
            src_cmp, dst_cmp, w = cheapest_edge[comp]

            # Needed because components might have been joined
            src_cmp = components.find(src_cmp)
            dst_cmp = components.find(dst_cmp)

            if src_cmp != dst_cmp:
                src_int_diff = internal_diff.get(src_cmp, 0)
                src_cmp_size = cmp_size.get(src_cmp, 1)
                src_diff = src_int_diff + (k / src_cmp_size)

                dst_int_diff = internal_diff.get(dst_cmp, 0)
                dst_cmp_size = cmp_size.get(dst_cmp, 1)
                dst_diff = dst_int_diff + (k / dst_cmp_size)

                if w <= min(src_diff, dst_diff):
                    components.union(src_cmp, dst_cmp)
                    n_components -= 1

                    internal_diff[src_cmp] = max(src_int_diff, w)
                    internal_diff[dst_cmp] = max(dst_int_diff, w)

                    cmp_size[src_cmp] = src_cmp_size + dst_cmp_size
                    cmp_size[dst_cmp] = src_cmp_size + dst_cmp_size


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

def segment_boruvka_3(image, k, min_cmp_size):
    edges = create_graph(image)
    m = len(edges)

    components = DisjointSet()
    internal_diff = dict()
    cmp_size = dict()

    n_rows = image.shape[0]
    n_cols = image.shape[1]

    prev_n_components = n_rows * n_cols + 1
    n_components = n_rows * n_cols

    while prev_n_components != n_components:
        prev_n_components = n_components
        cheapest_edge = dict()

        # Get smallest outgoing edge of each component that can be joined
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
                    if w <= cheapest_edge.get(src_cmp, (None, None, MAX_DIFF))[2]:
                        cheapest_edge[src_cmp] = (src, dst, w)

                    if w <= cheapest_edge.get(dst_cmp, (None, None, MAX_DIFF))[2]:
                        cheapest_edge[dst_cmp] = (src, dst, w)

        for key in cheapest_edge:
            src, dst, w = cheapest_edge[key]
            src_cmp = components.find(src)
            dst_cmp = components.find(dst)

            if src_cmp != dst_cmp:
                components.union(src_cmp, dst_cmp)
                n_components -= 1
                merged_cmp = components.find(src)
                internal_diff[merged_cmp] = max(internal_diff.get(merged_cmp, 0), w)

                src_cmp_size = cmp_size.get(src_cmp, 1)
                dst_cmp_size = cmp_size.get(dst_cmp, 1)
                cmp_size[merged_cmp] = src_cmp_size + dst_cmp_size

                rmv_cmp = src_cmp if merged_cmp == dst_cmp else dst_cmp
                if rmv_cmp in internal_diff:
                    del internal_diff[rmv_cmp]
                    del cmp_size[rmv_cmp]

    return components


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