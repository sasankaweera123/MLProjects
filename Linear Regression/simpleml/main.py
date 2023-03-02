from sklearn.linear_model import LinearRegression
import random


# Data set
def data_set():

    feature_set = []
    target_set = []

    n_rows = 200
    random_number_limit = 2000

    for i in range(0, n_rows):
        x = random.randint(0, random_number_limit)
        y = random.randint(0, random_number_limit)
        z = random.randint(0, random_number_limit)

        function = (10 * x) + (2 * y) + (3 * z)

        feature_set.append([x, y, z])
        target_set.append(function)

    return feature_set, target_set


# Model
def create_model():
    f_set, t_set = data_set()
    model = LinearRegression()
    model.fit(f_set, t_set)

    return model


# Test
def test_model(test):
    m = create_model()
    prediction = m.predict(test)
    print('Prediction:' + str(prediction) + '   Co - Efficient: ' + str(m.coef_))
    
    return prediction


# Main
if __name__ == '__main__':
    test_set = [[8, 4, 7]]
    test_model(test_set)
    
