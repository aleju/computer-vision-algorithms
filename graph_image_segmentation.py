from __future__ import division, print_function
from scipy import signal, ndimage
from scipy.ndimage import filters as filters
import numpy as np
import random
from skimage import data
from skimage import filters as skifilters
from skimage import util as skiutil
from skimage import feature
import util
from sklearn.decomposition import PCA
import os
from scipy import misc
from collections import defaultdict
np.random.seed(42)
random.seed(42)

K = 1.25

def main():
    im = data.camera()
    im = misc.imresize(im, (32, 32))
    height, width = im.shape
    nb_pixel = height * width

    nodes = dict()
    idx = 0
    for y in range(height):
        for x in range(width):
            nodes[(y, x)] = Node(idx, y, x, im[y, x])
            idx += 1

    edges = defaultdict(list)
    for y in range(height):
        for x in range(width):
            node = nodes[(y, x)]
            if y > 0:
                edges[(y, x)].append(Edge(node, nodes[(y-1, x)]))
            if x < (width-1):
                edges[(y, x)].append(Edge(node, nodes[(y, x+1)]))
            if y < (height-1):
                edges[(y, x)].append(Edge(node, nodes[(y+1, x)]))
            if x > 0:
                edges[(y, x)].append(Edge(node, nodes[(y, x-1)]))

    subgraphs = []
    for y in range(height):
        for x in range(width):
            subgraphs.append(SubGraph([nodes[(y, x)]], edges[(y, x)]))

    graph = Graph(subgraphs)

    converged = False
    while not converged:
        print("Merging (%d left)..." % (len(graph.subgraphs)))
        converged = True
        merged = []
        ediffs = graph.get_ediffs()
        for subgraph_a, subgraph_b, ediff in ediffs:
            if subgraph_a not in merged and subgraph_b not in merged:
                idiff_a = subgraph_a.get_internal_difference()
                idiff_b = subgraph_b.get_internal_difference()
                if ediff < K*idiff_a and ediff < K*idiff_b:
                    graph.merge_subgraphs(subgraph_a, subgraph_b)
                    merged.extend([subgraph_a, subgraph_b])
                    converged = False

    print("Found %d segments" % (len(graph.subgraphs)))

    subgraphs = sorted(graph.subgraphs, key=lambda sg: len(sg.nodes), reverse=True)
    nb_segments = 8
    segment_images = []
    segment_titles = []
    for i in range(min(nb_segments, len(subgraphs))):
        segment_titles.append("Segment %d/%d" % (i, len(subgraphs)))
        segment_image = np.zeros(im.shape)
        for node in subgraphs[i].nodes:
            segment_image[node.y, node.x] = 255
        segment_images.append(segment_image)

    images = [im]
    images.extend(segment_images)
    titles = ["Image"]
    titles.extend(segment_titles)
    util.plot_images_grayscale(images, titles)

class Node(object):
    def __init__(self, idx, y, x, value):
        self.idx = idx
        self.y = y
        self.x = x
        self.value = int(value)

class Edge(object):
    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = abs(from_node.value - to_node.value)

class SubGraph(object):
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.update_caches()

    def update_caches(self):
        self.node_ids = set([node.idx for node in self.nodes])

    def get_internal_difference(self):
        return np.average([edge.weight for edge in self.edges])

    def get_external_difference(self, to_subgraph):
        weights = []
        for edge in self.edges:
            if to_subgraph.contains_node(edge.to_node):
                weights.append(edge.weight)
        return min(weights) if len(weights) > 0 else 256

    def contains_node(self, node):
        #return any([my_node == node for my_node in self.nodes])
        return node.idx in self.node_ids

class Graph(object):
    def __init__(self, subgraphs):
        self.subgraphs = subgraphs

    def merge_subgraphs(self, subgraph_a, subgraph_b):
        subgraph_a.nodes.extend(subgraph_b.nodes)
        subgraph_a.edges.extend(subgraph_b.edges)
        subgraph_b.nodes = []
        subgraph_b.edges = []
        subgraph_a.update_caches()
        subgraph_b.update_caches()
        self.subgraphs = [subgraph for subgraph in self.subgraphs if subgraph != subgraph_b]

    """
    def find_lowest_ediff(self):
        best_pair = None
        best_diff = None
        for i in range(len(subgraphs)):
            for j in range(i+1, len(subgraphs)):
                diff = subgraphs[i].get_external_difference(subgraphs[j])
                if best_diff is None or best_diff > diff:
                    best_pair = (subgraphs[i], subgraphs[j])
                    best_diff = diff
        return best_pair, best_diff
    """

    def get_ediffs(self):
        ediffs = []
        for i in range(len(self.subgraphs)):
            for j in range(i+1, len(self.subgraphs)):
                ediff = self.subgraphs[i].get_external_difference(self.subgraphs[j])
                ediffs.append((self.subgraphs[i], self.subgraphs[j], ediff))
        ediffs = sorted(ediffs, key=lambda t: t[2])
        return ediffs

"""
class Graph(object):
    def __init__(self, pixel_coordinates, pixel_values, neighbors):
        self.pixel_coordinates = pixel_coordinates
        self.pixel_values = pixel_values
        self.neighbors = neighbors

    def add_neighbor(self, other_graph):
        self.neighbors.append(other_graph)

    def remove_neighbor(self, other_graph):
        self.neighbors = [neighbor for neighbor in self.neighbors if neighbor != other_graph]

    def get_internal_difference(self):
        return np.median(self.pixel_values)

    def get_external_differences(self):
        diffs = []
        for neighbor in self.neighbors:
            if len(neighbor.pixel_values) > 0:
                diff = math.min(abs(self.pixel_values[0] - neighbor.pixel_values[-1]))

    def merge_with(self, other_graph):
        other_neighbors = other_graph.neighbors
        for neighbor in other_neighbors:
            self.neighbors.append(neighbor)
            neighbor.add_neighbor(self)
            neighbor.remove_neighbor(other_graph)
        self.pixel_coordinates.extend(other_graph.pixel_coordinates)
        self.pixel_values.extend(other_graph.pixel_values)
        self.pixel_values = sorted(self.pixel_values)
        other_graph.pixel_coordinates = []
        other_graph.pixel_values = []

class ExternalDifferences(object):
    def __init__(self, diffs):
        self.diffs = diffs

    def get_minimal_diff(self):
        return self.diffs[0]

    def merge(self, graph_a, graph_b):
        for i in range(nb_graphs):
"""

if __name__ == "__main__":
    main()
