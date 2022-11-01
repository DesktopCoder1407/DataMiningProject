from hishiryo import Hishiryo
import matplotlib.pyplot as plt
import pandas

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


def plot_hishiryo():
    hishiryo_converter = Hishiryo.Hishiryo()
    output_path = "visualization/hishiryo.png"
    radius = 1000
    hishiryo_converter.convertCSVToRadialBitmap('data/cleaned_bank_churners.csv', ',', output_path, radius, ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
                            'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1'], 'Polygon')


plot_boxplots()
