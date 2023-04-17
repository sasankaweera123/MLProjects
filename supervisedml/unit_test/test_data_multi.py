import unittest
import data_multiFile


# test data_multiFile module using unittest
class TestDataMulti(unittest.TestCase):

    # test get_data in data_multiFile module
    def test_get_data(self):
        df = data_multiFile.pd.read_csv('../data/data_multi.csv')
        x, y = data_multiFile.get_data(df)
        self.assertEqual(x.shape, (13, 2))
        self.assertEqual(y.shape, (13,))

    # test get_stats in data_multiFile module
    def test_create_model(self):
        df = data_multiFile.pd.read_csv('../data/data_multi.csv')
        model = data_multiFile.create_model(df)
        self.assertEqual(model.coef_.shape, (2,))
        self.assertEqual(model.intercept_.shape, ())
        self.assertEqual(model.predict([[2300, 130]]).shape, (1,))

    # test plot_reg in data_multiFile module
    def test_plot_reg(self):
        df = data_multiFile.pd.read_csv('../data/data_multi.csv')
        data_multiFile.test_plot_reg(df, 'Weight')
        data_multiFile.test_plot_reg(df, 'Volume')

    # test plot_bar in data_multiFile module
    def test_plot_bar(self):
        df = data_multiFile.pd.read_csv('../data/data_multi.csv')
        data_multiFile.test_plot(df)


# run the test
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
