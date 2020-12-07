from disjoint_set import DisjointSet
import numpy as np
from scipy.ndimage import gaussian_filter
np.seterr(all='raise')
import math
import time
from operator import itemgetter

MAX_DIFF = math.sqrt(3 * pow(255,2))

def felzenswalb(input_image, sigma, k, min_cmp_size):
    # Apply gaussian kernel for smoothing, might be not necessary
    smoothed_image = gaussian_filter(input_image, sigma=sigma)

    start_time = time.time()

    V, E, W = create_graph_4_neigbour(smoothed_image)

    # Make sure operations on input image are floating point!
    segmentation_hierarchy = segment_fastmst_hierarchies(V, E, W)

    print("--- %s seconds ---" % (time.time() - start_time))
    return segmentation_hierarchy


def segment_fastmst_hierarchies(V, E, W):
    hierarchy = []

    # 15. Call the MST_Algorithm on the newly created graph until a single vertex remains
    # Could also save things to create hierarchy
    while (len(V)) > 1:
        V, E, W, supervertex_ids = MST_recursive(V, E, W)
        hierarchy.append(supervertex_ids)
        print(len(V))

    return hierarchy


def MST_recursive(V, E, W):
    # A. Find minimum weighted edge
    # - - - - - - - - - - - - - - -

    # 1. Append weight w and outgoing vertex v per edge into a list, X.
    # Normally 8-10 bit for weight, 20-22 bits for ID. Because of 32 bit limitation CUDPP scan primitive, probably not relevant anymore
    X = [el for el in zip(W, E)]  # in parallel for all edges

    # 2. Divide the edge-list, E, into segments with 1 indicating the start of each segment,
    #    and 0 otherwise, store this in flag array F.
    F = [0 for i in range(len(X))]  # in parallel for all edges
    for i in range(0, len(V)):  # in parallel for all vertices
        edges_start_idx = V[i]
        F[edges_start_idx] = 1

    # 3. Perform segmented min scan on X with F indicating segments
    #    to find minimum outgoing edge-index per vertex, store in NWE.
    NWE = []  # Index of mwoe

    min_edge_weight = math.inf
    min_edge_index = 0
    for i in range(len(X)):  # Using scan on O(E) elements
        edge_weight = X[i][0]
        if edge_weight < min_edge_weight:
            min_edge_weight = edge_weight
            min_edge_index = i

        if i + 1 == len(X) or F[i + 1] == 1:
            NWE.append(min_edge_index)
            min_edge_weight = math.inf

    # B. Finding and removing cycles
    # - - - - - - - - - - - - - - -

    # 4. Find the successor of each vertex and add to successor array, S.
    S = [X[min_edge_index][1] for min_edge_index in NWE]  # in parallel on all vertices

    # 5. Remove cycle making edges from NWE using S, and identify representatives vertices.
    for vertex, successor in enumerate(S):  # in parallel on all vertices
        successor_2 = S[successor]
        if vertex == successor_2:  # Cycle
            if vertex < successor:
                S[vertex] = vertex
            else:
                S[successor] = successor

    # C. Merging vertices and assigning IDs to supervertices
    # - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # 7. Propagate representative vertex ids using pointer doubling.
    change = True
    while change:
        change = False
        for vertex, successor in enumerate(S):  # For all vertices in parallel
            successor_2 = S[successor]
            if successor != successor_2:
                change = True
                S[vertex] = successor_2

    # 8. Append successor array’s entries with its index to form a list, L. Representative left, vertex id right, 64 bit
    L = [(representative, vertex) for vertex, representative in enumerate(S)]  # for all vertices in parallel

    # 9. Split L, create flag over split output and scan the flag to find new ids per vertex, store new ids in C.
    # 9.1 Split L using representative as key. In parallel using a split of O(V) with log(V) bit key size.
    #     Don't need sort in practice!
    L = sorted(L, key=lambda el: el[0])

    # 9.2 Create flag, first element not flagged so that can use simple sum for scan
    F2 = [0 for i in range(len(L))]  # Create flag to indicate boundaries, in parallel for all vertices
    for i in range(1, len(L)):  # in parallel for all vertices
        if L[i - 1][0] != L[i][0]:
            F2[i] = 1

    # 9.3 Scan flag to assign new IDs, Using a scan on O(V) elements
    C = []
    cur_id = 0
    for i in range(0, len(L)):
        cur_id += F2[i]
        C.append(cur_id)

    # D. Removing self edges
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Find supervertex ids of u and v for each edge using C
    # 10.1 Create mapping from each original vertex ID to its new supervertex ID so we can lookup supervertex IDs directly
    supervertex_ids = [0 for i in range(0, len(L))]
    for i in range(0, len(L)):  # in parallel for all vertices
        vertex_id = L[i][1]
        supervertex_id = C[i]
        supervertex_ids[vertex_id] = supervertex_id

    # 10.2 Create vector indicating source vertex u for each edge
    F[0] = 0
    u_ids = []
    cur_id = 0
    for i in range(0, len(F)):  # in parallel for all edges
        cur_id += F[i]
        u_ids.append(cur_id)

    # 11. Remove edge from edge-list if u, v have same supervertex id (remove self edges)
    for i in range(0, len(E)):  # in parallel for all edges
        id_u = u_ids[i]
        supervertexid_u = supervertex_ids[id_u]

        id_v = E[i]
        supervertexid_v = supervertex_ids[id_v]

        if supervertexid_u == supervertexid_v:
            E[i] = math.inf  # Mark edge for removal

    # E. Removing duplicate edges
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # 12. Remove the largest duplicate edges using split over new u,v and w.
    # 12.1 Append supervertex ids of u and v along with index i into single 64 bit array (u 24 bit, v 24 bit, i x bit)
    UVI = []
    for i in range(0, len(E)):  # in parallel for all edges
        id_u = u_ids[i]
        id_v = E[i]
        index = i
        if id_v != math.inf:
            supervertexid_u = supervertex_ids[id_u]
            supervertexid_v = supervertex_ids[id_v]
            UVI.append((supervertexid_u, supervertexid_v, index))
        else:
            # Could also change both E[i] and id_u[i] to point to supervertex with infinite ID to avoid else
            UVI.append((math.inf, math.inf, index))  # Inf so at back when splitting?

    # 12.1 Split UVI by sorting first on u.supervertexid, then v.supervertexid
    # Can prob be done better, first get sorted indices, then restructure array using these indices
    sorted_indices = [i[0] for i in sorted(enumerate(UVI), key=lambda x: x[1])]  # in parallel
    UVI = itemgetter(*sorted_indices)(UVI)

    # 12.1.1 Create a flag indicating the start of each run of parallel edges
    F5 = [0 for i in range(len(E))]  # in parallel for all edge
    F5[0] = 1
    new_edge_size = len(E)

    for i in range(1, len(E)):  # in parallel for all edges
        supervertexid_u_prev, supervertexid_v_prev, prev_idx = UVI[i-1]
        supervertexid_u, supervertexid_v, idx = UVI[i]

        if supervertexid_u != math.inf and supervertexid_v != math.inf:
            if supervertexid_u_prev != supervertexid_u or supervertexid_v_prev != supervertexid_v:
                F5[i] = 1
        else:
            new_edge_size = min(new_edge_size, i)  # I guess we need sort for this to work, also needed for next step

    # 12.3 Create flag indicating smallest edges by perform segmented min scan on UVI with F5 indicating start of
    #    duplicate edge runs. 0 for larger duplicates
    min_duplicate_idx = []
    min_edge_weight = math.inf
    min_edge_index = 0

    # From now can create new kernel size of new edge size for next operations to ignore duplicate edges between components (set to infinity)

    for i in range(new_edge_size):  # in parallel for all edges
        supervertexid_u, supervertexid_v, idx = UVI[i]
        edge_weight = W[idx]

        if edge_weight < min_edge_weight:
            min_edge_weight = edge_weight
            min_edge_index = i

        if i + 1 == new_edge_size or F5[i + 1] == 1:
            min_duplicate_idx.append(min_edge_index)
            min_edge_weight = math.inf

    F3 = [0 for i in range(new_edge_size)]  # in parallel for all edges
    for i in range(len(min_duplicate_idx)):
        idx = min_duplicate_idx[i]
        F3[idx] = 1

    # 13. Compact and create new edge and weight list
    # 13.1 Scan flag to get location min entries in new edge list
    cur_location = -1  # So starts from 0
    compact_locations = []
    for i in range(0, new_edge_size):  # in parallel for all edges
        cur_location += F3[i]
        compact_locations.append(cur_location)

    # New edge list etc.
    new_E = [0 for i in range(new_edge_size)]
    new_W = [0 for i in range(new_edge_size)]

    expanded_u = [0 for i in range(new_edge_size)]  # Used for creating new vertex list

    # 13.2 Compact and create new edge and weight list
    new_E_size = 0
    new_V_size = 0
    for i in range(0, new_edge_size):  # In parallel for all edges (can use new edge size)
        if F3[i]:
            supervertex_id_u, supervertex_id_v, index = UVI[i]
            edge_weight = W[index]
            new_location = compact_locations[i]
            if supervertex_id_u != math.inf and supervertex_id_v != math.inf:  # Shouldn't be necessary I think if sorted TODO check
                new_E[new_location] = supervertex_id_v
                new_W[new_location] = edge_weight
                expanded_u[new_location] = supervertex_id_u

                new_E_size = max(new_location + 1, new_E_size)
                new_V_size = max(supervertex_id_v + 1, new_V_size)

    # 13.3 Resize lists to actual size (probably not necessary in C as long as pass on list length)
    remove_tail = new_edge_size - new_E_size
    if remove_tail > 0:
        new_E = new_E[:-remove_tail]
        new_W = new_W[:-remove_tail]
        expanded_u = expanded_u[:-remove_tail]

    # Can again use new kernel size here for actual edge list size

    # 14. Build the vertex list from the newly formed edge list
    # 14.1 Create flag based on difference in u on the new edge list
    F4 = [0 for i in range(new_E_size)]
    if new_E_size > 0:
        F4[0] = 1
    for i in range(1, new_E_size):  # in parallel for all edges
        if expanded_u[i - 1] != expanded_u[i]:
            F4[i] = 1

    # 14.2 Build the vertex list from the newly formed edge list
    new_V = [0 for i in range(new_V_size)]
    for i in range(new_E_size):  # in parallel for all edges
        if F4[i] == 1:
            id_u = expanded_u[i]
            new_V[id_u] = i

    return new_V, new_E, new_W, supervertex_ids

def dissimilarity(image, row1, col1, row2, col2):
    return float(math.sqrt(
        np.sum(
            pow(image[row1][col1] - image[row2][col2], 2)
        )
    ))


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