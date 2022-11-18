from preprocessing import build_PCA
import datetime
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time

data = build_PCA(16)

x = data.drop(columns=['Attrition_Flag'])
y = data['Attrition_Flag']

model = Sequential([
    Dense(32, activation='relu', input_shape = (16,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.AUC(),
                       tf.keras.metrics.BinaryAccuracy()])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=0)

start_time = time.time()

model.fit(x_train, y_train, 
          epochs=250, validation_split = .1, 
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./logs/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '/'),
                     tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

training_time = time.time() - start_time
metrics = model.evaluate(x_test, y_test)

for name, metric in zip(model.metrics_names, metrics):
    print(f'{name}: {metric:.4f}')
print(f'It took {training_time / 60} minutes to train the model.')