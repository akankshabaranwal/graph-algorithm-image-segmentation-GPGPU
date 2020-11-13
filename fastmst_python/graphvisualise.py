import networkx as nx
import matplotlib.pyplot as plt

class GraphVisualization:

    def __init__(self):
        self.graph = nx.Graph()

    def fromAdjacency(self, V, E, W):
        for u, start in enumerate(V):
            until = len(E) if u + 1 == len(V) else V[u+1]
            for idx in range(start, until):
                self.addEdge(u, E[idx], W[idx])

    def addMST(self, V, E, MST):
        for u, start in enumerate(V):
            until = len(E) if u + 1 == len(V) else V[u+1]
            for idx in range(start, until):
                if MST[idx]:
                    self.graph[u][E[idx]]['thickness'] = 3

    # edge and appends it to the visual list
    def addEdge(self, a, b, weight):
        self.graph.add_edge(a, b, thickness=1, weight=weight)
        # In visualize function G is an object of

    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        #thickness = nx.get_edge_attributes(self.graph, 'thickness').values()

        pos=nx.circular_layout(self.graph)

        thickness = [self.graph[u][v]['thickness'] for u, v in self.graph.edges]

        nx.draw(self.graph, pos, width=thickness)
        weights = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=weights)


        #nx.draw_networkx(self.graph, width=list(thickness), edge_labels=list(weights))
        plt.show()