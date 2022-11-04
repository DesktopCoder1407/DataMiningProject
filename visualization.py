from hishiryo import Hishiryo
import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
from sklearn.preprocessing import StandardScaler

data = pandas.read_csv('data/cleaned_bank_churners.csv')


def plot_boxplots():
    plotted_data = [('Customer_Age', 'Years'), ('Dependent_count', ''), ('Total_Relationship_Count', ''),
                    ('Contacts_Count_12_mon', ''), ('Months_on_book', 'Months'), ('Months_Inactive_12_mon', 'Months'),
                    ('Credit_Limit', 'Dollars'), ('Avg_Open_To_Buy', 'Dollars'), ('Total_Revolving_Bal', 'Dollars'),
                    ('Total_Trans_Amt', 'Dollars'), ('Total_Trans_Ct', '')]

    for item in plotted_data:
        plt.boxplot(data[item[0]])
        plt.xticks([1], [item[0]])
        plt.ylabel(item[1])
        #plt.show()
        plt.savefig('visualization/' + item[0] + '.png')
        plt.close()


def plot_hishiryo():
    hishiryo_converter = Hishiryo.Hishiryo()
    output_path = "visualization/hishiryo.png"
    radius = 1000
    hishiryo_converter.convertCSVToRadialBitmap('data/cleaned_bank_churners.csv', ',', output_path, radius, ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
                            'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1'], 'Polygon')


def plot_heatmap():
    data = pandas.read_csv('data/cleaned_bank_churners.csv')
    features = ['Customer_Age', 'Dependent_count', 'Total_Relationship_Count', 'Contacts_Count_12_mon',
                'Months_on_book', 'Months_Inactive_12_mon', 'Credit_Limit', 'Avg_Open_To_Buy',
                'Total_Revolving_Bal', 'Total_Trans_Amt', 'Total_Trans_Ct']

    data = pandas.DataFrame(StandardScaler().fit_transform(data[features]), columns=features)
    covariance_matrix = numpy.cov(data.transpose(), bias=True)

    fig = plt.figure(figsize=(14, 10))
    seaborn.heatmap(covariance_matrix, annot=True, xticklabels=features, yticklabels=features)
    # plt.show()
    plt.savefig('visualization/heatmap.png')
    plt.close()


plot_heatmap()
