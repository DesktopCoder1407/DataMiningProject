from preprocessing import build_PCA
from keras import Sequential
from keras.layers import Dense
import pandas
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = pandas.read_csv('data/cleaned_bank_churners.csv')
x = data.drop(columns=['Attrition_Flag'])
y = data['Attrition_Flag']

model = Sequential([
    Dense(32, activation='relu', input_shape = (16,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=.05),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.AUC(curve='PR'),
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.BinaryAccuracy()])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=0)

# Do model.fit() here to fit training data to model. Afterwords: evaluate model's performance.