import unittest
import Advertising

# test Advertising module using unittest
class TestAdvertising(unittest.TestCase):

    # test get_data in Advertising module
    def test_get_data(self):
        df = Advertising.pd.read_csv('../data/Advertising.csv')
        x, y = Advertising.get_data(df)
        self.assertEqual(x.shape, (200, 3))
        self.assertEqual(y.shape, (200,))

    # test get_stats in Advertising module
    def test_create_model(self):
        df = Advertising.pd.read_csv('../data/Advertising.csv')
        model = Advertising.create_model(df)
        self.assertEqual(model.coef_.shape, (3,))
        self.assertEqual(model.intercept_.shape, ())
        self.assertEqual(model.predict([[2300, 130, 0]]).shape, (1,))

    # test plot_reg in Advertising module
    def test_plot_reg(self):
        df = Advertising.pd.read_csv('../data/Advertising.csv')
        Advertising.test_plot_reg(df)

    # test plot_bar in Advertising module
    def test_plot_bar(self):
        df = Advertising.pd.read_csv('../data/Advertising.csv')
        Advertising.test_plot_bar(df)

# run the test
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
