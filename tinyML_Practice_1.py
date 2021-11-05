import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

start_Time = time.time()

"""Setup layers and neurons"""
my_layer_1 = tf.keras.layers.Dense(units=2, input_shape=[1])
my_layer_2 = tf.keras.layers.Dense(units=1)
"""Sequential layers | compile model with mean squared error"""
model = tf.keras.Sequential([my_layer_1,my_layer_2])
model.compile(optimizer='sgd', loss='mean_squared_error')

"""Training dataset"""
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

"""Build model"""
model.fit(xs, ys, epochs=500)

"""Print learned weights and biases for each neuron"""
print(my_layer_1.get_weights())
print(my_layer_2.get_weights())

"""Test input"""
value_Predict = 10.0

"""Model prediction"""
model_Value = model.predict([value_Predict])
print("model predcition: {}".format(model_Value))

print("time: {}".format(time.time()-start_Time))

"""Manually compute the answer"""
"""Assign neuron w and b for layer 1"""
layer_1_w1 = (my_layer_1.get_weights()[0][0][0])
layer_1_w2 = (my_layer_1.get_weights()[0][0][1])
layer_1_b1 = (my_layer_1.get_weights()[1][0])
layer_1_b2 = (my_layer_1.get_weights()[1][1])

"""Assign neuron w and b for layer 2"""
layer_2_w1 = (my_layer_2.get_weights()[0][0])
layer_2_w2 = (my_layer_2.get_weights()[0][1])
layer_2_b = (my_layer_2.get_weights()[1][0])

n1_output = (layer_1_w1*value_Predict) + layer_1_b1
n2_output = (layer_1_w2*value_Predict) + layer_1_b2

"""Sum and report prediction/output"""
n3_output = (layer_2_w1*n1_output) + (layer_2_w2*n2_output) + layer_2_b

print(n3_output)

