from preprocessing import build_PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Loading data
data = build_PCA(16)

x = data.drop(columns=['Attrition_Flag'])
y = data['Attrition_Flag']

# Create feature and target arrays
X = data.data
y = data.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

# Predict on dataset which model has not seen before
print(knn.predict(X_test))

