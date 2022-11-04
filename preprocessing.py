import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def clean_data():
    data = pandas.read_csv('data/bank_churners.csv')
    data = data[data['Education_Level'] != 'Unknown']
    data = data[data['Marital_Status'] != 'Unknown']
    data = data[data['Income_Category'] != 'Unknown']
    data.drop(columns=['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                       'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
              inplace=True)

    data.to_csv('data/cleaned_bank_churners.csv', index=False)


def build_PCA(num_components):
    data = pandas.read_csv('data/cleaned_bank_churners.csv')
    original_data = pandas.read_csv('data/cleaned_bank_churners.csv')
    data.drop(columns=['CLIENTNUM', 'Attrition_Flag'], inplace=True)

    features = ['Customer_Age', 'Dependent_count', 'Total_Relationship_Count', 'Contacts_Count_12_mon',
                'Months_on_book', 'Months_Inactive_12_mon', 'Credit_Limit', 'Avg_Open_To_Buy',
                'Total_Revolving_Bal', 'Total_Trans_Amt', 'Total_Trans_Ct']
    x = data.loc[:, features].values
    data = StandardScaler().fit_transform(x)

    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(data)
    principal_data = pandas.DataFrame(data=principal_components,
                                      columns=['Principal Component ' + str(x + 1) for x in range(num_components)])

    print(f'Explained Variance per Component: {pca.explained_variance_ratio_}')
    print(f'Sum of Explained Variance: {sum(pca.explained_variance_ratio_)}')

    final_data = pandas.concat([principal_data, original_data['Attrition_Flag']], axis=1)
    return final_data


def display_PCA():
    data = build_PCA(3)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_zlabel('Principal Component 3', fontsize=12)
    ax.set_title('3 Component PCA', fontsize=20)
    targets = ['Existing Customer', 'Attrited Customer']
    colors = ['r', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = data['Attrition_Flag'] == target
        ax.scatter(data.loc[indicesToKeep, 'Principal Component 1'],
                   data.loc[indicesToKeep, 'Principal Component 2'],
                   data.loc[indicesToKeep, 'Principal Component 3'],
                   c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()
    plt.savefig('visualization/PCA.png')
    plt.close()


build_PCA(8)
display_PCA()
