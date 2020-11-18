import numpy as np
import math
import networkx as nx
from operator import itemgetter

import numpy as np
import math
import networkx as nx
from operator import itemgetter

# Remarks
# - Is X created and used for storage savings? Since max edge weight 442 = 9 bits have 23 bits left for vertices. This
#   could give problems if have more than 8.388.608 vertices. Also restricts us to integer calculation instead of floats
#   should check result of using integers instead of floats on segmentation results.
# - Quoted from paper: "Duplicate edge elimination is an optional operation because the MST algorithm works correctly
#   even in the presence of multiple edges. However, the initial iterations usually have a large number of multiple
#   edges, thus increasing the length of the edge list and making all edge-based operations slower. We therefore remove
#   duplicate edges in initial operations. Later iterations do not require this step because the edge-list size
#   decreases with iteration number."
# - Can probably reuse some arrays here and there to save memory
# - For the compressed adjacency list format, always add edges in both directions for an undirected graph.
# - Split primitive: Do we need to sort for this?  I think paper says can be implemented linearly in amount of elements.
#   Just brings together all elements with same key should be enough maybe, but bucket sort is also linear.

def MST_recursive(V, E, W, orig_edge, MST):
    
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

        if i + 1 == len(X) or F[i+1] == 1:
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

    # 6. Mark remaining edges from NWE as part of output in MST.
    for vertex, successor in enumerate(S):  # in parallel on all vertices
        if vertex != successor:  # All edges except from representative part of MST
            vertex_min_edge_idx = NWE[vertex]
            orig_edge_id = orig_edge[vertex_min_edge_idx]
            MST[orig_edge_id] = 1

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
    # 12.1 Append supervertex ids of u and v along with weight w into single 64 bit array (u 24 bit, v 24 bit, w 16 bit)
    UVW = []
    for i in range(0, len(E)):  # in parallel for all edges
        id_u = u_ids[i]
        id_v = E[i]
        edge_weight = W[i]
        if id_v != math.inf:
            supervertexid_u = supervertex_ids[id_u]
            supervertexid_v = supervertex_ids[id_v]
            UVW.append((supervertexid_u, supervertexid_v, edge_weight))
        else:
            # Could also change both E[i] and id_u[i] to point to supervertex with infinite ID to avoid else
            UVW.append((math.inf, math.inf, edge_weight)) # Inf so at back when splitting?

    # 12.2 Split UVW by sorting first on u.supervertexid, then v.supervertexid, then weight
    # Can prob be done better, first get sorted indices, then restructure array using these indices
    # I guess we need sort here so min weight first, maybe it would suffice to have only w sorted when u and v same
    sorted_indices = [i[0] for i in sorted(enumerate(UVW), key=lambda x: x[1])] # in parallel
    UVW = itemgetter(*sorted_indices)(UVW)

    # 12.3 Create flag indicating smallest edges, 0 for larger duplicates (first entry sorted UVW is smallest if duplicate)
    F3 = [0 for i in range(len(UVW))]  # in parallel for all edges
    F3[0] = 1
    new_edge_size = len(E)
    for i in range(1, len(UVW)):  # in parallel for all edges
        prev_supervertexid_u = UVW[i - 1][0]
        prev_supervertexid_v = UVW[i - 1][1]

        supervertexid_u = UVW[i][0]
        supervertexid_v = UVW[i][1]

        if supervertexid_u != math.inf and supervertexid_v != math.inf:
            if prev_supervertexid_u != supervertexid_u or prev_supervertexid_v != supervertexid_v: # If not sorted need to use or
                F3[i] = 1
        else:
            new_edge_size = min(new_edge_size, i) # I guess we need sort for this to work, also needed for next step

    # From now can create new kernel size of new edge size for next operations to ignore duplicate edges between components (set to infinity)

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
    new_orig_edge = [0 for i in range(new_edge_size)]

    expanded_u = [0 for i in range(new_edge_size)]  # Used for creating new vertex list

    # 13.2 Compact and create new edge and weight list
    new_E_size = 0
    new_V_size = 0
    for i in range(0, new_edge_size):  # In parallel for all edges (can use new edge size)
        if F3[i]:
            supervertex_id_u, supervertex_id_v, edge_weight = UVW[i]
            new_location = compact_locations[i]
            if supervertex_id_u != math.inf and supervertex_id_v != math.inf:  # Shouldn't be necessary I think if sorted TODO check
                new_E[new_location] = supervertex_id_v
                new_W[new_location] = edge_weight
                expanded_u[new_location] = supervertex_id_u

                # Store original edge id of each edge so can mark MST edges at right position in next iterations
                orig_edge_pos = sorted_indices[i]
                new_orig_edge[new_location] = orig_edge[orig_edge_pos]

                new_E_size = max(new_location+1, new_E_size)
                new_V_size = max(supervertex_id_v+1, new_V_size)

    # 13.3 Resize lists to actual size (probably not necessary in C as long as pass on list length)
    remove_tail = new_edge_size - new_E_size
    if remove_tail > 0:
        new_E = new_E[:-remove_tail]
        new_W = new_W[:-remove_tail]
        expanded_u = expanded_u[:-remove_tail]
        new_orig_edge = new_orig_edge[:-remove_tail]

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

    return new_V, new_E, new_W, new_orig_edge, MST


def compute_MST(V, E, W):
    # Mapping from edge id to orig edge id (for later recursive iterations when edges changed)
    # gives edge ID in original input E for an edge ID in a current recursive iteration
    orig_edge = [i for i in range(0, len(E))]

    MST = [0 for i in range(len(E))]  # bitmap edges E present in MST, 1 if edge in MST, else 0

    # 15. Call the MST_Algorithm on the newly created graph until a single vertex remains
    # Could also save things to create hierarchy
    while (len(V)) > 1:
        V, E, W, orig_edge, MST = MST_recursive(V, E, W, orig_edge, MST)

    return MST


def MST_visualise(V, E, W):
    G = GraphVisualization()
    G.fromAdjacency(V, E, W)

    # Copy because else get changed
    input_V = V[:]
    input_E = E[:]
    input_W = W[:]
    MST = compute_MST(input_V, input_E, input_W)

    G.addMST(V, E, MST)
    G.visualize()


def MST_test(V, E, W):

    import networkx
    G = GraphVisualization()
    G.fromAdjacency(V, E, W)

    grand_truth = nx.minimum_spanning_tree(G.graph)

    # Copy because else get changed
    input_V = V[:]
    input_E = E[:]
    input_W = W[:]
    MST = compute_MST(input_V, input_E, input_W)

    MST_edges = []
    for u, start in enumerate(V):
        until = len(E) if u + 1 == len(V) else V[u + 1]
        for idx in range(start, until):
            if MST[idx]:
                v = E[idx]
                if u < v:
                    MST_edges.append((u, v))
                else:
                    MST_edges.append((v, u))

    grand_truth_edges = []
    for u, v in grand_truth.edges():
        if u < v:
            grand_truth_edges.append((u, v))
        else:
            grand_truth_edges.append((v, u))

    print("MST edges:          {}".format(sorted(MST_edges)))
    print("Expected MST edges: {}".format(sorted(grand_truth_edges)))
    print("Outputs match: {}".format(sorted(MST_edges) == sorted(grand_truth_edges)))


if __name__ == '__main__':
    # Graph: compressed adjacency list
    V = [0, 3, 7, 12, 16, 20, 25, 31, 35, 39]
    E = [4, 2, 1, 0, 2, 6, 7, 0, 4, 3, 6, 1, 2, 4, 5, 6, 0, 2, 3, 5, 4, 3, 6, 8, 9, 1, 2, 3, 5, 8, 7, 9, 8, 6, 1, 9, 7, 6, 5, 5, 8, 7]
    W = [6, 3, 9, 9, 9, 8, 18, 3, 4, 2, 9, 9, 2, 2, 9, 8, 6, 4, 2, 9, 9, 9, 7, 5, 4, 8, 9, 8, 7, 9, 10, 4, 3, 10, 18, 1, 3, 9, 5, 4, 1, 4]

    # Other test graph
    #V = [0, 2, 4, 7, 9, 11]
    #E = [4, 1, 5, 0, 3, 5, 4, 2, 5, 0, 2, 3, 2, 1]
    #W = [9, 40, 11, 40, 4, 3, 10, 4, 56, 9, 10, 56, 3, 11]

    MST_test(V, E, W)
    #MST_visualise(V, E, W)



