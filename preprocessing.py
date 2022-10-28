import pandas

data = pandas.read_csv('data/bank_churners.csv')
data = data[data['Education_Level'] != 'Unknown']
data = data[data['Marital_Status'] != 'Unknown']
data = data[data['Income_Category'] != 'Unknown']
data.drop(columns=['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                   'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
          inplace=True)

data.to_csv('data/cleaned_bank_churners.csv', index=False)