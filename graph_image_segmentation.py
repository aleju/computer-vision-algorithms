"""Segment an image with a simplified graph method based on internal differences
(within subgraphs/segments) and external differences (between subgraphs/segments).
Each pixel initially is a subgraph with edges to its 4 neighbors. Subgraphs
where externalDifference < k*internalDifference are merged until convergence."""
from __future__ import division, print_function
import numpy as np
import random
from skimage import data
import util
from scipy import misc
from collections import defaultdict
np.random.seed(42)
random.seed(42)

# Parameter to tune when two subgraphs/segments will be merged.
# The higher the value, the more often subgraphs will be merged.
K = 1.25

def main():
    """Load image, convert into graph, segment."""
    img = data.camera()
    # worked decently for 32x32, not so well and slow for 64x64
    img = misc.imresize(img, (32, 32))

    graph = image_to_graph(img)

    # Merge subgraphs until convergence
    converged = False
    while not converged:
        print("Merging (%d subgraphs left)..." % (len(graph.subgraphs)))
        converged = True
        merged = []
        ediffs = graph.get_ediffs()
        # Iterate over all pairs subgraphs that have a connecting edge
        # Merge them if possible.
        # Do not stop after the first merge, so that we don't have to
        # recalculate the external differences between subgraphs many times
        for subgraph_a, subgraph_b, ediff in ediffs:
            # Try to merge the subgraphs to one subgraph
            if subgraph_a not in merged and subgraph_b not in merged:
                idiff_a = subgraph_a.get_internal_difference()
                idiff_b = subgraph_b.get_internal_difference()
                # Merge if externel difference (between two subgraphs)
                # is below both internal differences
                if ediff < K*idiff_a and ediff < K*idiff_b:
                    graph.merge_subgraphs(subgraph_a, subgraph_b)
                    merged.extend([subgraph_a, subgraph_b])
                    converged = False

    print("Found %d segments" % (len(graph.subgraphs)))

    # Create images of segments
    subgraphs = sorted(graph.subgraphs, key=lambda sg: len(sg.nodes), reverse=True)
    nb_segments = 8
    segment_images = []
    segment_titles = []
    for i in range(min(nb_segments, len(subgraphs))):
        segment_titles.append("Segment %d/%d" % (i, len(subgraphs)))
        segment_image = np.zeros(img.shape)
        for node in subgraphs[i].nodes:
            segment_image[node.y, node.x] = 255
        segment_images.append(segment_image)

    # plot images
    images = [img]
    images.extend(segment_images)
    titles = ["Image"]
    titles.extend(segment_titles)
    util.plot_images_grayscale(images, titles)

def image_to_graph(img):
    """Converts an image to a graph of segments."""
    height, width = img.shape

    # Create nodes of graph
    nodes = dict()
    idx = 0
    for y in range(height):
        for x in range(width):
            nodes[(y, x)] = Node(idx, y, x, img[y, x])
            idx += 1

    # Create edges
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

    # Create subgraphs (segments), one per pixel
    subgraphs = []
    for y in range(height):
        for x in range(width):
            subgraphs.append(SubGraph([nodes[(y, x)]], edges[(y, x)]))

    # Create graph
    graph = Graph(subgraphs)

    return graph

class Node(object):
    """Nodes (=Pixel) in the graph."""
    def __init__(self, idx, y, x, value):
        """Create node for a pixel with a node-index, xy-coordinates and a pixel
        intensity value."""
        self.idx = idx
        self.y = y
        self.x = x
        self.value = int(value)

class Edge(object):
    """Edges between nodes."""
    def __init__(self, from_node, to_node):
        """Create a new edge between two nodes (=pixels). Edge weight is the
        intensity difference between the pixels."""
        self.from_node = from_node
        self.to_node = to_node
        self.weight = abs(from_node.value - to_node.value)

class SubGraph(object):
    """Subgraphs in the graph (aka segments)."""
    def __init__(self, nodes, edges):
        """Create a new subgraph/segment with nodes and edges."""
        self.nodes = nodes
        self.edges = edges
        self.update_caches()

    def update_caches(self):
        """Update the node id cache of the subgraph (ids of nodes that this
        subgraph contains)."""
        self.node_ids = set([node.idx for node in self.nodes])

    def get_internal_difference(self):
        """Get the internal difference within the subgraph. It is simply the
        average of the edge weights."""
        return np.average([edge.weight for edge in self.edges])

    def get_external_difference(self, to_subgraph):
        """Get the external difference between this subgraph and another
        subgraph. It is the minimum edge weight between the subgraphs, or
        256 if there is no connecting edge."""
        weights = []
        for edge in self.edges:
            if to_subgraph.contains_node(edge.to_node):
                weights.append(edge.weight)
        return min(weights) if len(weights) > 0 else 256

    def contains_node(self, node):
        """Returns whether a node is contained within this subgraph."""
        return node.idx in self.node_ids

class Graph(object):
    """Graph containing all subgraphs (i.e. image that contains segments)."""
    def __init__(self, subgraphs):
        """Create a new graph containing several subgraphs."""
        self.subgraphs = subgraphs

    def merge_subgraphs(self, subgraph_a, subgraph_b):
        """Merge two subgraphs/segments to one subgraph/segment. Automatically
        extends subgraph A and removes subgraph B."""
        subgraph_a.nodes.extend(subgraph_b.nodes)
        subgraph_a.edges.extend(subgraph_b.edges)
        subgraph_b.nodes = []
        subgraph_b.edges = []
        subgraph_a.update_caches()
        subgraph_b.update_caches()
        self.subgraphs = [subgraph for subgraph in self.subgraphs if subgraph != subgraph_b]

    def get_ediffs(self):
        """Calculate the external differences of all pairs of subgraphs within
        the graph. This is an expensive operation, especially before the first
        merges."""
        ediffs = []
        for i in range(len(self.subgraphs)):
            for j in range(i+1, len(self.subgraphs)):
                ediff = self.subgraphs[i].get_external_difference(self.subgraphs[j])
                ediffs.append((self.subgraphs[i], self.subgraphs[j], ediff))
        ediffs = sorted(ediffs, key=lambda t: t[2])
        return ediffs

if __name__ == "__main__":
    main()
