import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

my_layer_1 = tf.keras.layers.Dense(units=2, input_shape=[1])
my_layer_2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([my_layer_1,my_layer_2])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

 

model.fit(xs, ys, epochs=500)

print(my_layer_1.get_weights())
print(my_layer_2.get_weights())

value_Predict = 10.0

layer_1_w1 = (my_layer_1.get_weights()[0][0][0])
layer_1_w2 = (my_layer_1.get_weights()[0][0][1])
layer_1_b1 = (my_layer_1.get_weights()[1][0])
layer_1_b2 = (my_layer_1.get_weights()[1][1])


layer_2_w1 = (my_layer_2.get_weights()[0][0])
layer_2_w2 = (my_layer_2.get_weights()[0][1])
layer_2_b = (my_layer_2.get_weights()[1][0])

n1_output = (layer_1_w1*value_Predict) + layer_1_b1
n2_output = (layer_1_w2*value_Predict) + layer_1_b2

n3_output = (layer_2_w1*n1_output) + (layer_2_w2*n2_output) + layer_2_b

print(n3_output)