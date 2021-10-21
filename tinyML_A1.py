import tensorflow as tf
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.python.keras.layers.core import Activation


"""set up variables and functions"""
predictions = []
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        predictions.append(model.predict(xs))
callbacks = myCallback()

"""define xs and ys"""
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

Shape = [1]
Loss = 'mean_squared_error'

# model = keras.Sequential([keras.layers.Dense(units=8, input_shape=Shape)])
model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = Shape)])
model.compile(optimizer='sgd',loss=Loss)

"""fit model"""
model.fit(xs,ys,epochs=250, callbacks=[callbacks],verbose=2)

epoch_numbers = [1,25,50,100,250]
plt.plot(xs,ys,label='Ys')
for epoch in epoch_numbers:
    plt.plot(xs,predictions[epoch-1],label='Epoch = '+str(epoch))
plt.legend()
plt.show()

print(model.predict([10]))
