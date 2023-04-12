import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns


def get_data(data):
    x = data[['Weight', 'Volume']]
    y = data['CO2']
    return x, y


def create_model(data):
    x, y = get_data(data)
    reg_model = linear_model.LinearRegression()
    reg_model.fit(x.values, y)

    return reg_model


def test_plot(data):
    x, y = get_data(data)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.barplot(x=x['Weight'], y=y, ax=ax[0])
    sns.barplot(x=x['Volume'], y=y, ax=ax[1])
    plt.tight_layout()
    plt.show()


def test_plot_reg(data, value):
    x, y = get_data(data)
    sns.regplot(x=x[value], y=y)
    plt.show()


def main():
    df = pd.read_csv('data/data_multi.csv')
    model = create_model(df)
    print(model.coef_)
    print(model.intercept_)

    prediction = model.predict([[2300, 130]])
    print(prediction)

    # test_plot(df)
    test_plot_reg(df, 'Weight')
    test_plot_reg(df, 'Volume')


if __name__ == '__main__':
    main()
