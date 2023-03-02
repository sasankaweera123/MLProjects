import unittest
import main


# Test class
class TestMain(unittest.TestCase):

    # Test data set
    def test_data_set(self):
        f_set, t_set = main.data_set()
        self.assertEqual(len(f_set), 200)
        self.assertEqual(len(t_set), 200)

    # Test model
    def test_create_model(self):
        model = main.create_model()
        self.assertIsNotNone(model)

    # Test model
    def test_test_model(self):
        test_set = [[8, 4, 7]]
        prediction = main.test_model(test_set)
        self.assertIsNotNone(prediction)


if __name__ == '__main__':
    unittest.main()
