from DataPreProcessing import X, y, IMG_SIZE, x_test, y_test
from keras import models
from keras import layers
from keras.optimizers import SGD
from keras.applications import InceptionV3


Incp_con = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
model = models.Sequential()
model.add(Incp_con)

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(7, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy",
               optimizer=sgd,
               metrics=['accuracy']
             )
model.fit(X, y, epochs=200, validation_split=0.2, batch_size=32)
model.save("model.h5")
model.evaluate(x_test, y_test)