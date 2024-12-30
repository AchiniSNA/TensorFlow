import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import numpy as np

import tensorflow as tf
from tensorflow import keras     # Keras is a software library(an interface) use to build ANNs
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train) , (x_test, y_test) = mnist.load_data()

print(x_train.shape)    #(60000, 28, 28)
print(y_train.shape)    #(60000,)

x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0     # -1 means 60000 remains as it is. An image consist 28*28 pixels. here we reshape and normalize the data.
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

#x_train = tf.convert_to_tensor(x_train)
#x_test = tf.convert_to_tensor(x_test)

def createModel():
    model = keras.Sequential()       # Used sequential api
    model.add(keras.Input(shape=(28*28,)))     # Input layer
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10))        # Output layer
    return model

mymodel = createModel()
print(mymodel.summary())

'''
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 512)                 │         401,920 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 256)                 │         131,328 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 10)                  │           2,570 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 535,818 (2.04 MB)
 Trainable params: 535,818 (2.04 MB)
 Non-trainable params: 0 (0.00 B)
'''



mymodel.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

mymodel.fit(x_train, y_train,batch_size=32,epochs=5,verbose=2)
mymodel.evaluate(x_test,y_test,batch_size=32, verbose=2)

'''
Epoch 1/5
1875/1875 - 14s - 7ms/step - accuracy: 0.9436 - loss: 0.1879
Epoch 2/5
1875/1875 - 8s - 4ms/step - accuracy: 0.9758 - loss: 0.0783
Epoch 3/5
1875/1875 - 8s - 5ms/step - accuracy: 0.9825 - loss: 0.0553
Epoch 4/5
1875/1875 - 8s - 4ms/step - accuracy: 0.9869 - loss: 0.0414
Epoch 5/5
1875/1875 - 9s - 5ms/step - accuracy: 0.9890 - loss: 0.0344
313/313 - 1s - 3ms/step - accuracy: 0.9782 - loss: 0.0774
'''