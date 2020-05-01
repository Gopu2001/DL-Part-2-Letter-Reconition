import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
import random
import matplotlib.pyplot as plt
from emnist import extract_training_samples, extract_test_samples

x_train, y_train = extract_training_samples('letters')
x_test, y_test = extract_test_samples('letters')

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(27, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, validation_split=0.45, epochs=1000, batch_size=32,
          verbose=2, callbacks=[EarlyStopping(monitor='accuracy', min_delta=0.001, mode='auto')])

model.evaluate(x_test, y_test, verbose=0)
model.save("id_char.h5")

'''
image_index = random.randint(0,9999)
plt.imshow(x_test[image_index].reshape(28,28), cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
plt.show()
'''
