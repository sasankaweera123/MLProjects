import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model


def data_cleaning(df):
    # drop Months Code 7021
    df = df[df['Months Code'] != 7021]
    # replace value LCU with 1 and SLC with 0
    df.loc[:, 'Element Code'] = df['Element Code'].replace(['LCU', 'SLC'], [1, 0])
    return df


def get_data(df):
    x = df[['Element Code', 'Year Code', 'Months Code']]
    y = df['Value']
    x = x.dropna()
    y = y.drop(0)
    # print(y)
    # print(x.head())
    return x, y


def create_model(df):
    x, y = get_data(df)
    model = linear_model.LinearRegression()
    model.fit(x.values, y)
    return model


def plot_data(df):
    x, y = get_data(df)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.barplot(x=x['Element Code'], y=y, ax=ax[0])
    sns.barplot(x=x['Year Code'], y=y, ax=ax[1])
    plt.tight_layout()
    plt.show()


def test_plot(df, x_value):
    x, y = get_data(df)
    sns.scatterplot(x=x[x_value], y=y)
    plt.show()


def main():
    df = pd.read_csv("data/exchange-rates_lka.csv")
    # print(df)
    df = data_cleaning(df)
    # get_data(df)
    model = create_model(df)
    print(model.coef_)
    print(model.intercept_)
    # plot_data(df)
    # test_plot(df, 'Months Code')

    prediction = model.predict([[1, 2019, 7007]])
    print(prediction)



if __name__ == "__main__":
    main()
