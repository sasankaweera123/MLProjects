import unittest
import data_lin


# test data_lin module using unittest
class TestDataLin(unittest.TestCase):

    # test get_data in data_lin module
    def test_get_data(self):
        df = data_lin.pd.read_csv('../data/data_lin.csv')
        x, y = data_lin.get_data(df)
        self.assertEqual(x.shape, (13,))
        self.assertEqual(y.shape, (13,))

    # test get_stats in data_lin module
    def test_get_stats(self):
        df = data_lin.pd.read_csv('../data/data_lin.csv')
        x, y = data_lin.get_data(df)
        slope, intercept, r_value, p_value, std_err = data_lin.get_stats(x, y)
        self.assertEqual(slope, -1.7512877115526118)
        self.assertEqual(intercept, 103.10596026490066)
        self.assertEqual(r_value, -0.758591524376155)
        self.assertEqual(p_value, 0.002646873922456106)
        self.assertEqual(std_err, 0.453536157607742)

    # test function in data_lin module
    def test_function(self):
        self.assertEqual(data_lin.function(6, -1.7512877115526118, 103.10596026490066), 92.59823399558499)

    # test plot in data_lin module
    def test_plot(self):
        df = data_lin.pd.read_csv('../data/data_lin.csv')
        x, y = data_lin.get_data(df)
        slope, intercept, r_value, p_value, std_err = data_lin.get_stats(x, y)
        data_lin.plot(x, y, slope, intercept)


# run the test
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
