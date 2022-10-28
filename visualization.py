from hishiryo import Hishiryo
import matplotlib.pyplot as plt
import pandas

data = pandas.read_csv('data/cleaned_bank_churners.csv')

data_dict = {'Customer_Age': data['Customer_Age'], 'Dependent_count': data['Dependent_count'],
             'Months_on_book': data['Months_on_book'], 'Total_Relationship_Count': data['Total_Relationship_Count'],
             'Months_Inactive_12_mon': data['Months_Inactive_12_mon'], 'Contacts_Count_12_mon': data['Contacts_Count_12_mon'],
             'Credit_Limit': data['Credit_Limit'], 'Total_Revolving_Bal': data['Total_Revolving_Bal'],
             'Avg_Open_To_Buy': data['Avg_Open_To_Buy'], 'Total_Trans_Amt': data['Total_Trans_Amt'],
             'Total_Trans_Ct': data['Total_Trans_Ct']}

# Years: Customer_Age
# Small_Numeric: Dependent_count, Total_Relationship_Count, Contacts_Count_12_mon
# Months: Months_on_book, Months_Inactive_12_mon
# Money: Credit_Limit, Avg_Open_To_Buy
# Balance: Total_Revolving_Bal, Total_Trans_Amt
# Transaction Count: Total_Trans_Ct

# Years
plt.boxplot(data['Customer_Age'])
plt.xticks([1], ['Customer_Age'])
plt.ylabel('Years')
plt.show()

# Small_Numeric
fig, ax = plt.subplots()
ax.boxplot([data['Dependent_count', 'Total_Relationship_Count', 'Contacts_Count_12_mon']])
ax.set_xticklabels(['Dependent_count', 'Total_Relationship_Count', 'Contacts_Count_12_mon'])
plt.ylabel('')
plt.show()

def plot_hishiryo():
    hishiryo_converter = Hishiryo.Hishiryo()
    output_path = "visualization/hishiryo.png"
    radius = 1000
    hishiryo_converter.convertCSVToRadialBitmap('data/cleaned_bank_churners.csv', ',', output_path, radius, ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
                            'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1'], 'Polygon')
