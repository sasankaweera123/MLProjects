import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# 1. Read the data from the csv file
def get_data(data):
    features = data['age']
    label = data['speed']

    return features, label


# 2. Calculate the slope, intercept, r_value, p_value, std_err
def get_stats(features, label):
    slope, intercept, r_value, p_value, std_err = stats.linregress(features, label)
    return slope, intercept, r_value, p_value, std_err


# 3. Create a function that uses the slope and intercept to predict Y
def function(x, slope, intercept):
    return slope * x + intercept


# 4. Plot the data and the prediction
def plot(features, label, slope, intercept):
    plt.scatter(features, label)
    plt.plot(features, slope * features + intercept, 'r')
    plt.show()


def main():
    path = pd.read_csv('data/data_lin.csv')
    features, label = get_data(path)
    slope, intercept, r_value, p_value, std_err = get_stats(features, label)

    # Y = 103.10596026490066 + -1.7512877115526118 * X
    # r = -0.758591524376155
    # p = 0.002646873922456106

    print("Y = " + str(intercept) + " + " + str(slope) + "X")
    print("r = " + str(r_value))
    print("p = " + str(p_value))

    # 5. Predict the speed of aged 6
    print(function(6, slope, intercept))

    # 6. Plot the data and the prediction
    plot(features, label, slope, intercept)


# 7. Run the program
if __name__ == '__main__':
    main()
