import unittest
import main
import numpy as np


class TestMain(unittest.TestCase):

    # test the Node class
    def test_node(self):
        node = main.Node([3, 4], 0)
        self.assertEqual(node.data, [3, 4])
        self.assertEqual(node.axis, 0)
        self.assertEqual(node.left, None)
        self.assertEqual(node.right, None)

    # test the KDTree class root node axis
    def test_kdtree_one(self):
        points = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
        tree = main.KDTree(points)
        self.assertEqual(tree.root.axis, 0)

    # test the KDTree class root node data
    def test_kdtree_two(self):
        points = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
        tree = main.KDTree(points)
        expected_root = np.array([7, 2])
        self.assertTrue(np.array_equal(tree.root.data, expected_root))

    # test the KDTree class find_nearest_neighbor method
    def test_kdtree_three(self):
        points = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
        tree = main.KDTree(points)
        self.assertTrue(np.array_equal(tree.find_nearest_neighbor(np.array([3, 4.5])), [2, 3]))


if __name__ == '__main__':
    unittest.main()