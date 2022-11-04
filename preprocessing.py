import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def clean_data():
    data = pandas.read_csv('data/bank_churners.csv')

    # Remove any unknown values
    data = data[data['Education_Level'] != 'Unknown']
    data = data[data['Marital_Status'] != 'Unknown']
    data = data[data['Income_Category'] != 'Unknown']

    # Drop Naive Bayes Categories (Recommended by Dataset)
    data.drop(columns=['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                       'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
              inplace=True)

    # Drop Irrelevant Data
    data.drop(columns=['CLIENTNUM'])

    data.to_csv('data/cleaned_bank_churners.csv', index=False)


def build_PCA(num_components, return_PCA = False):
    data = pandas.read_csv('data/cleaned_bank_churners.csv')

    x = data.loc[:, data.dtypes != object].values
    x = StandardScaler().fit_transform(x)  # TODO: May have to scale numeric differently than categorical?

    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(x)
    principal_data = pandas.DataFrame(data=principal_components,
                                      columns=['Principal Component ' + str(x + 1) for x in range(num_components)])
    for i in range(len(pca.explained_variance_ratio_)):
        print(f'Explained Variance for Component {i + 1}: {pca.explained_variance_ratio_[i]:.2%}')
    print(f'Sum of Explained Variance: {sum(pca.explained_variance_ratio_):.2%}')

    final_data = pandas.concat([principal_data, data['Attrition_Flag']], axis=1)
    if return_PCA:
        return pca
    return final_data


def graph_PCA_variance(num_components):
    pca = build_PCA(num_components, return_PCA=True)

    # Data
    x = [x+1 for x in range(num_components)]
    y = [x*100 for x in pca.explained_variance_ratio_]

    # Line and Scatter Plot
    plt.plot(x, y)
    plt.scatter(x, y)

    # Display the data label near the point & have X and Y labels
    for i, label in enumerate(y):
        plt.annotate(f'{round(label, 1)}%', (x[i], y[i]))
    plt.xlabel('PCA Component')
    plt.ylabel('Explained Variance in %')
    plt.title(f'Sum of Explained Variance: {sum(pca.explained_variance_ratio_):.2%}')

    plt.show()
    plt.close()


def display_PCA():
    data = build_PCA(3)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Title and Labels
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_zlabel('Principal Component 3', fontsize=12)
    ax.set_title('3 Component PCA', fontsize=20)

    # Create two groups: one for each class
    targets = ['Existing Customer', 'Attrited Customer']
    colors = ['r', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = data['Attrition_Flag'] == target
        ax.scatter(data.loc[indicesToKeep, 'Principal Component 1'],
               data.loc[indicesToKeep, 'Principal Component 2'],
               data.loc[indicesToKeep, 'Principal Component 3'],
               c=color, s=4)
    ax.legend(targets)

    # Set the format as a grid and display/save
    ax.grid()
    plt.show()
    plt.savefig('visualization/PCA.png')
    plt.close()


# build_PCA(8)
# display_PCA()
graph_PCA_variance(8)

# TODO: Determine the types of Categorical data (Make separate sheet to show datatypes for each column). Determine how to switch the categorical data to numeric.
# data = pandas.read_csv('data/cleaned_bank_churners.csv')
# data = data.loc[:, data.dtypes == object]
#
# print(data.loc[0])
