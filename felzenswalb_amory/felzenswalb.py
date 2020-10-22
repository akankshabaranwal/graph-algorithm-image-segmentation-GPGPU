from disjoint_set import DisjointSet
from graph import Graph
import numpy as np

def create_image_graph(image):
    n_rows = image.shape[0]
    n_cols = image.shape[1]

    # Create graph
    g = Graph()
    for i in range(0, n_rows - 1):
        for j in range(0, n_cols - 1):
            cur_node = i * n_cols + j
            cur_node_intensity = int(image[i][j])

            right_node = i * n_cols + j + 1
            right_node_intensity = int(image[i][j+1])

            bottom_node = (i + 1) * n_cols + j
            bottom_node_intensity = int(image[i+1][j])

            diag_down_node = (i + 1) * n_cols + j + 1
            diag_down_node_intensity = int(image[i+1][j + 1])

            g.addEdge(cur_node, right_node, abs(cur_node_intensity - right_node_intensity))
            g.addEdge(cur_node, bottom_node, abs(cur_node_intensity - bottom_node_intensity))
            g.addEdge(cur_node, diag_down_node, abs(cur_node_intensity - diag_down_node_intensity))

    for i in range(1, n_rows):  # Add diagonal up right edges
        for j in range(0, n_cols - 1):
            cur_node = i * n_cols + j
            cur_node_intensity = int(image[i][j])

            diag_up_node = (i - 1) * n_cols + j + 1
            diag_up_node_intensity = int(image[i - 1][j + 1])

            g.addEdge(cur_node, diag_up_node, abs(cur_node_intensity - diag_up_node_intensity))

    for i in range(0, n_rows - 1):  # Add right column edges
        j = n_cols - 1

        cur_node = i * n_cols + j
        cur_node_intensity = int(image[i][j])

        bottom_node = (i + 1) * n_cols + j
        bottom_node_intensity = int(image[i + 1][j])

        g.addEdge(cur_node, bottom_node, abs(cur_node_intensity - bottom_node_intensity))

    for j in range(0, n_cols - 1):  # Add bottom row edges
        i = n_rows - 1

        cur_node = i * n_cols + j
        cur_node_intensity = int(image[i][j])

        right_node = i * n_cols + j + 1
        right_node_intensity = int(image[i][j + 1])

        g.addEdge(cur_node, right_node, abs(cur_node_intensity - right_node_intensity))

    return g

def felzenswalb(image, k):
    ds = DisjointSet()
    internal_diff = dict()
    # internal_diff.get("key", 0)
    cmp_size = dict()
    # cmp_size.get("key", 1)

    g = create_image_graph(image)
    g.sortEdges()

    m = len(g.graph)
    for q in range(0, m):
        src, dst, w = g.graph[q]
        src_cmp = ds.find(src)
        dst_cmp = ds.find(dst)
        if src_cmp != dst_cmp:
            src_int_diff = internal_diff.get(src_cmp, 0)
            src_cmp_size = cmp_size.get(src_cmp, 1)
            src_diff = src_int_diff + (k / src_cmp_size)

            dst_int_diff = internal_diff.get(dst_cmp, 0)
            dst_cmp_size = cmp_size.get(dst_cmp, 1)
            dst_diff = dst_int_diff + (k / dst_cmp_size)
            if w <= min(src_diff, dst_diff):
                # Merge components
                ds.union(src_cmp, dst_cmp)
                merged_cmp = ds.find(src)
                internal_diff[merged_cmp] = max(src_int_diff, dst_int_diff)
                cmp_size[merged_cmp] = src_cmp_size + dst_cmp_size

                rmv_cmp = src_cmp if merged_cmp == dst_cmp else dst_cmp
                if rmv_cmp in internal_diff:
                    del internal_diff[rmv_cmp]
                    del cmp_size[rmv_cmp]

    return ds