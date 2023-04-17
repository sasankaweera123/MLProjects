import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns


# ge the relevant  data from data set
def get_data(data):
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    return x, y


# create ML model
def create_model(data):
    x, y = get_data(data)
    reg_model = linear_model.LinearRegression()
    reg_model.fit(x.values, y)

    return reg_model


# Create Plot in order to have an idea about data set
def test_plot_reg(data):
    x, y = get_data(data)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.regplot(x=x['TV'], y=y, ax=ax[0])
    sns.regplot(x=x['Radio'], y=y, ax=ax[1])
    sns.regplot(x=x['Newspaper'], y=y, ax=ax[2])
    plt.tight_layout()  # to make sure that all the plots are visible
    plt.show()


# Create Plot in order to have an idea about data set
def test_plot_bar(data):
    x, y = get_data(data)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.barplot(x=x['TV'], y=y, ax=ax[0])
    sns.barplot(x=x['Radio'], y=y, ax=ax[1])
    sns.barplot(x=x['Newspaper'], y=y, ax=ax[2])
    plt.tight_layout()
    plt.show()


# Main Function that have basic function need to be run
def main():
    df = pd.read_csv('data/Advertising.csv')
    # print(df.head())
    model = create_model(df)
    print(model.coef_)
    print(model.intercept_)
    print(model.predict([[2300, 130, 0]]))  # test the model using test data

    # test_plot_reg(df)
    # test_plot_bar(df)


# Run the main function
if __name__ == '__main__':
    main()
