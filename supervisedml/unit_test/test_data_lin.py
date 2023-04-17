import unittest
import Advertising


class TestAdvertising(unittest.TestCase):

    def test_get_data(self):
        df = Advertising.pd.read_csv('../data/Advertising.csv')
        x, y = Advertising.get_data(df)
        self.assertEqual(x.shape, (200, 3))
        self.assertEqual(y.shape, (200,))

    def test_create_model(self):
        df = Advertising.pd.read_csv('../data/Advertising.csv')
        model = Advertising.create_model(df)
        self.assertEqual(model.coef_.shape, (3,))
        self.assertEqual(model.intercept_.shape, ())
        self.assertEqual(model.predict([[2300, 130, 0]]).shape, (1,))

    def test_plot_reg(self):
        df = Advertising.pd.read_csv('../data/Advertising.csv')
        Advertising.test_plot_reg(df)
    
    def test_plot_bar(self):
        df = Advertising.pd.read_csv('../data/Advertising.csv')
        Advertising.test_plot_bar(df)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
