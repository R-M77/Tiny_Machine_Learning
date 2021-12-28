"""Lesson_1_sample_tf_code"""
# import tensorflow as tf

# class MyModel(tf.keras.Model):
#     def __init__(self):
#         super(MyModel,self).__init__()
#         self.conv = tf.keras.layers.Conv2D(32,3, activation='ralu')
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense1 = tf.keras.layers.Dense(128, activation='ralu')
#         self.dense2 = tf.keras.layers.Dense(10)

#         def call(self,x):
#             x = self.conv(x)
#             x = self.flatten(x)
#             x = self.dense1(x)
#             x = self.dense2(x)
#             return x
# model = MyModel()

"""Lesson_2_manual_linear_regression"""
# # Import lib
# from __future__ import absolute_import, division, print_function, unicode_literals
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# # initialize guess
# initial_w = 10.0
# initial_b = 10.0

# # define loss function | sqr error
# def loss(predicted_y, target_y):
#     return tf.reduce_mean(tf.square(predicted_y-target_y))

# # define training
# def train(model, inputs, outputs, learning_rate):
#     with tf.GradientTape() as t:
#         current_loss = loss(model(inputs), outputs)
#         # differentiate model values wrt loss
#         dw, db = t.gradient(current_loss, [model.w, model.b])
#         # update model based on learning_rate
#         model.w.assign_sub(learning_rate*dw)
#         model.b.assign_sub(learning_rate*db)

#         return current_loss

# # define model | linear regression
# class Model(object):
#     def __init__(self):
#         # initialize weights
#         self.w = tf.Variable(initial_w)
#         self.b = tf.Variable(initial_b)

#     def __call__(self,x):
#         return self.w *x +self.b

# # define inputs and learning_rate
# xs = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
# ys = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
# step_size = 0.09

# # initialize model
# model = Model()

# # collect values of w and b | previous values
# list_w, list_b = [],[]
# epochs = range(50)
# losses = []
# for epoch in epochs:
#     list_w.append(model.w.numpy())
#     list_b.append(model.b.numpy())
#     current_loss = train(model,xs,ys,learning_rate=step_size)
#     losses.append(current_loss)

#     print('Epoch {}: w={} b={}, loss={}'.format(epoch,list_w[-1],list_b[-1],current_loss))

# # plot w and b values for each training epoch vs true values
# TRUE_w = 2.0
# TRUE_b = -1.0
# plt.plot(epochs,list_w,'r',epochs,list_b,'b')
# plt.plot([TRUE_w]*len(epochs),'r--',[TRUE_b]*len(epochs), 'b--')
# plt.legend(['w','b','True_w','True_b'])
# plt.show()

# """Lesson_3_NN_linear-regression"""
# import tensorflow as tf
# import numpy as np
# from tensorflow import keras
# import math

# model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# model.compile(optimizer='sgd',loss='mean_squared_error')

# xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],dtype=float)
# ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0],dtype=float)

# model.fit(xs,ys,epochs=500)

# print(model.predict([10.0]))

"""Lesson_4_Minimizing_Loss_Gradient_Tape"""
from numpy.core.arrayprint import _leading_trailing
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

initial_W = 10.0
initial_B = 10.0

# Define loss function
def loss(predicted_y, target_y):
    return tf.reduce_mean(tf.square(predicted_y-target_y))

def train(model,inputs,outputs,learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs),outputs)
        dw,db = t.gradient(current_loss,[model.w,model.b])
        model.w.assign_sub(learning_rate*dw)
        model.b.assign_sub(learning_rate*db)
        return current_loss

# Define linear regression model
class Model(object):
    def __init__(self):
        self.w = tf.Variable(initial_W)
        self.b = tf.Variable(initial_B)

    def __call__(self, x):
        return self.w * x + self.b
    
xs = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
ys = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
LR = 0.09

model = Model()
list_w, list_b = [],[]
epochs = range(50)
losses = []
for epoch in epochs:
    list_w.append(model.w.numpy())
    list_b.append(model.b.numpy())
    current_loss = train(model,xs,ys,learning_rate=LR)
    losses.append(current_loss)
    print('Epoch {}: w={}, b={}, loss={}'.format(epoch,list_w[-1],list_b[-1],current_loss))

True_w = 2.0
True_b = -1.0
plt.plot(epochs,list_w,'r', epochs,list_b,'b')
plt.plot([True_w]*len(epochs),'r--',[True_b]*len(epochs),'b--')
plt.legend()
plt.show()