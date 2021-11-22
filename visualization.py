import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


"""visualize data"""
def show_images(img,val_images):
    plt.figure()
    plt.imshow(val_images[img].reshape(28,28))
    plt.grid(False)
    plt.axis('off')
    plt.show()

f, axarr = plt.subplots(3,2)
first_image = 2
second_image = 3
third_image = 5

convolution_number = 1
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs = layer_outputs)

for x in range(0,2):
    f1 = activation_model.predict(val_images[first_image].reshape(1,28,28,1))[x]
    axarr[0,x].imshow(f1[0,:,:,convolution_number],cmap = 'inferno')
    axarr[0,x].grid(False)

    f2 = activation_model.predict(val_images[second_image].reshape(1,28,28,1))[x]
    axarr[1,x].imshow(f2[0,:,:,convolution_number],cmap = 'inferno')
    axarr[1,x].grid(False)

    f3 = activation_model.predict(val_images[third_image].reshape(1,28,28,1))[x]
    axarr[2,x].imshow(f3[0,:,:,convolution_number],cmap = 'inferno')
    axarr[2,x].grid(False)

show_images(first_image,val_images)
show_images(second_image,val_images)
show_images(third_image,val_images)