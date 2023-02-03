import numpy as np


# Node class for KDTree
class Node:
    def __init__(self, data, axis, left=None, right=None):
        self.data = data
        self.axis = axis
        self.left = left
        self.right = right


# KDTree class for finding nearest neighbor in a set of points in a 2d space
class KDTree:
    def __init__(self, data_points):
        self.root = self.build_tree(data_points, 0)

    # Builds a KDTree from a set of data points
    def build_tree(self, data_points, axis):
        if len(data_points) == 0:
            return None
        data_points = sorted(data_points, key=lambda x: x[axis])
        median = len(data_points) // 2
        return Node(data_points[median], axis, self.build_tree(data_points[:median], (axis + 1) % 2), self.build_tree(data_points[median + 1:], (axis + 1) % 2))

    # Finds the nearest neighbor to a given point
    def find_nearest_neighbor(self, point):
        return self.find_nearest_neighbor_helper(self.root, point, self.root.data, 0)

    # Helper function for finding the nearest neighbor to a given point
    def find_nearest_neighbor_helper(self, node, point, nearest_neighbor, axis):
        if node is None:
            return nearest_neighbor
        if np.linalg.norm(point - node.data) < np.linalg.norm(point - nearest_neighbor):
            nearest_neighbor = node.data
        if point[axis] < node.data[axis]:
            nearest_neighbor = self.find_nearest_neighbor_helper(node.left, point, nearest_neighbor, (axis + 1) % 2)
            if np.linalg.norm(point - nearest_neighbor) > abs(point[axis] - node.data[axis]):
                nearest_neighbor = self.find_nearest_neighbor_helper(node.right, point, nearest_neighbor, (axis + 1) % 2)
        else:
            nearest_neighbor = self.find_nearest_neighbor_helper(node.right, point, nearest_neighbor, (axis + 1) % 2)
            if np.linalg.norm(point - nearest_neighbor) > abs(point[axis] - node.data[axis]):
                nearest_neighbor = self.find_nearest_neighbor_helper(node.left, point, nearest_neighbor, (axis + 1) % 2)
        return nearest_neighbor


# Test code for KDTree class
if __name__ == '__main__':
    points = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    tree = KDTree(points)
    print(tree.find_nearest_neighbor(np.array([3, 4.5])))
