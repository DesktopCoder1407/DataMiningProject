from preprocessing import build_PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import time


# Loading data
data = build_PCA(16)

x = data.drop(columns=['Attrition_Flag'])
y = data['Attrition_Flag']

# Create feature and target arrays
#X = data.values
#y = data.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state=42)

neighbors = np.arange(1, 19)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    start_time = time.time()
    knn.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

    # Output accuracy and training time for each K value
    print(f'''For K value of {k}:
    Accuracy: {test_accuracy[i]:.4f}
    Training Time: {training_time:.2f} seconds''')

# Output average accuracy for all K values
print(f'Mean Testing Accuracy is {sum(test_accuracy)/len(test_accuracy):.4f}')

# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.savefig('visualization/knearest.png')
plt.close()
#plt.show()