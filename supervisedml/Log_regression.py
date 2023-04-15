import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report


def test_plot(data):
    plt.rc('font', size=14)
    sns.set(style='white')
    sns.set(style='whitegrid', color_codes=True)
    sns.countplot(x='y', data=data, palette='hls')
    plt.show()


def test_plot_job(data):
    pd.crosstab(data.job, data.y).plot(kind='bar')
    plt.title('Purchase Frequency for Job Title')
    plt.xlabel('Job')
    plt.ylabel('Frequency of Purchase')
    plt.tight_layout()
    plt.show()


def test_plot_marital(data):
    table = pd.crosstab(data.marital, data.y)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Chart of Marital Status vs Purchase')
    plt.xlabel('Marital Status')
    plt.ylabel('Proportion of Customers')
    plt.tight_layout()
    plt.show()


def test_plot_education(data):
    table = pd.crosstab(data.education, data.y)
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Chart of Education vs Purchase')
    plt.xlabel('Education')
    plt.ylabel('Proportion of Customers')
    plt.tight_layout()
    plt.show()


def test_plot_day_of_week(data):
    pd.crosstab(data.day_of_week, data.y).plot(kind='bar')
    plt.title('Purchase Frequency for Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Frequency of Purchase')
    plt.tight_layout()
    plt.show()


def test_plot_month(data):
    pd.crosstab(data.month, data.y).plot(kind='bar')
    plt.title('Purchase Frequency for Month')
    plt.xlabel('Month')
    plt.ylabel('Frequency of Purchase')
    plt.tight_layout()
    plt.show()


def test_plot_age(data):
    data.age.hist()
    plt.title('Histogram of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def test_plot_p_outcome(data):
    pd.crosstab(data.poutcome, data.y).plot(kind='bar')
    plt.title('Purchase Frequency for P_outcome')
    plt.xlabel('P_outcome')
    plt.ylabel('Frequency of Purchase')
    plt.tight_layout()
    plt.show()


def get_percentage_y(data):
    y_no_count = len(data[data['y'] == 0])
    y_yes_count = len(data[data['y'] == 1])
    percentage_no = y_no_count / (y_no_count + y_yes_count)
    percentage_yes = y_yes_count / (y_no_count + y_yes_count)
    print('percentage of no: ', percentage_no * 100)
    print('percentage of yes: ', percentage_yes * 100)


def searching_complement(data):
    cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                'poutcome']
    for var in cat_vars:
        cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(data[var], prefix=var)
        data1 = data.join(cat_list)
        data = data1

    cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                'poutcome']
    data_vars = data.columns.values.tolist()
    to_keep = [i for i in data_vars if i not in cat_vars]
    data_final = data[to_keep]
    # print(data_final.columns.values)

    return data_final


def split_data(data):
    x = data.loc[:, data.columns != 'y']
    y = data.loc[:, data.columns == 'y']
    return x, y


def oversampling(x, y):
    os = SMOTE(random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    columns = x_train.columns
    os_data_x, os_data_y = os.fit_resample(x_train, y_train)
    os_data_x = pd.DataFrame(data=os_data_x, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])
    return os_data_x, os_data_y


def test_over_sampling(x, y):
    print("length of oversampled data is ", len(x))
    print("Number of no subscription in oversampled data", len(y[y['y'] == 0]))
    print("Number of subscription", len(y[y['y'] == 1]))
    print("Proportion of no subscription data in oversampled data is ", len(y[y['y'] == 0]) / len(x))
    print("Proportion of subscription data in oversampled data is ", len(y[y['y'] == 1]) / len(x))


def logistic_regression(data_Final, os_data_x, os_data_y):
    data_Final_vars = data_Final.columns.values.tolist()
    y = ['y']
    x = [i for i in data_Final_vars if i not in y]
    logit_model = LogisticRegression(solver='liblinear')
    rfe = RFE(logit_model, n_features_to_select=20, step=1)
    rfe = rfe.fit(os_data_x, os_data_y.values.ravel())
    print(rfe.support_)
    print(rfe.ranking_)


def composed_minneapolis(os_data_x, os_data_y):
    cols = ['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'month_apr',
            'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct',
            'poutcome_failure', 'poutcome_success']
    x = os_data_x[cols]
    y = os_data_y['y']
    logit_model = sm.Logit(y, x)
    result = logit_model.fit()
    # print(result.summary2())
    return result


def turkish_matters(os_data_x, os_data_y):
    x_train, x_test, y_train, y_test = train_test_split(os_data_x, os_data_y, test_size=0.3, random_state=0)
    logreg = LogisticRegression(solver='liblinear')
    logreg.fit(x_train, y_train.values.ravel())
    y_pred = logreg.predict(x_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
    print(confusion_matrix(y_test, y_pred))


def graphical_plots(os_data_x, os_data_y):
    x_train, x_test, y_train, y_test = train_test_split(os_data_x, os_data_y, test_size=0.3, random_state=0)
    logreg = LogisticRegression(solver='liblinear')
    logreg.fit(x_train, y_train.values.ravel())
    y_pred = logreg.predict(x_test)
    cm_display = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=logreg.classes_)
    cm_display.plot()
    plt.show()

def classification_report(os_data_x, os_data_y):
    x_train, x_test, y_train, y_test = train_test_split(os_data_x, os_data_y, test_size=0.3, random_state=0)
    logreg = LogisticRegression(solver='liblinear')
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    print(classification_report(y_test, y_pred))

def main():
    df = pd.read_csv('data/bank.csv', header=0)
    df = df.dropna()
    # print(df['education'].unique())
    # print(df['y'].value_counts())
    # test_plot(df)
    # get_percentage_y(df)
    # test_plot_job(df)
    # test_plot_marital(df)
    # test_plot_education(df)
    # test_plot_day_of_week(df)
    # test_plot_month(df)
    # test_plot_age(df)
    # test_plot_p_outcome(df)

    data_final = searching_complement(df)
    x, y = split_data(data_final)
    os_data_x, os_data_y = oversampling(x, y)
    # test_over_sampling(os_data_x, os_data_y)
    # Error Check needed
    # logistic_regression(data_final, os_data_x, os_data_y)
    # composed_minneapolis(os_data_x, os_data_y) Error
    # turkish_matters(os_data_x, os_data_y)
    graphical_plots(os_data_x, os_data_y)
    # classification_report(os_data_x, os_data_y) Error

if __name__ == '__main__':
    main()
