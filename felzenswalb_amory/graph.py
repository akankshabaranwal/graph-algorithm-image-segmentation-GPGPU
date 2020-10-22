class Graph:

    def __init__(self):
        self.graph = []  # default dictionary

    def addEdge(self, src, dst, weight):
        self.graph.append([src, dst, weight])

    def sortEdges(self):
        self.graph = sorted(self.graph, key=lambda el: el[2])

