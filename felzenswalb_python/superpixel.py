
# Use 4 neigbourhood

# Creates compressed adjacency list graph based on 4 neigbourhood
# Edges created twice in both directions
# o - o
# |   |
# o - o
def create_graph_4_neigbour(image):
    n_rows = image.shape[0]
    n_cols = image.shape[1]

    vertices = []
    edges = []

    # Weights can change in this algorithm so calculated from rgb values
    weights = [] # Weight for each item in edge array

    cur_edge_idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            left_node = i * n_cols + j - 1
            right_node = i * n_cols + j + 1
            bottom_node = (i+1) * n_cols + j
            top_node = (i - 1) * n_cols + j

            # Create pointer to start of edge array
            vertices.append(cur_edge_idx)

            if j > 0:
                edges.append(left_node)
                weights.append(dissimilarity(image, i, j, i, j - 1))
                cur_edge_idx += 1

            if j < n_cols - 1:
                edges.append(right_node)
                weights.append(dissimilarity(image, i, j, i, j + 1))
                cur_edge_idx += 1

            if i < n_rows - 1:
                edges.append(bottom_node)
                weights.append(dissimilarity(image, i, j, i + 1, j))
                cur_edge_idx += 1

            if i > 0:
                edges.append(top_node)
                weights.append(dissimilarity(image, i, j, i - 1, j))
                cur_edge_idx += 1

    return (vertices, edges, weights)