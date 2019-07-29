import tensorflow as tf
import numpy as np

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/(g.sum())

batch_size = 16
dimension1 = 64
dimension2 = 64
color_dimension = 3

blur_kernel = fspecial_gauss(size=5, sigma=2.5)
blur_kernel_repeat = blur_kernel.reshape((5, 5, 1, 1))
blur_kernel_repeat = np.repeat(blur_kernel_repeat, color_dimension, axis=2)
blur_kernel_tensor = tf.constant(blur_kernel_repeat, dtype=tf.float32)

std = 0.01

def cs_X_XT(input):
    store_shape = tf.shape(input)
    runner = tf.matmul(cs_tensor, tf.transpose(tf.layers.flatten(input)))
    return tf.reshape(tf.transpose(runner), shape=store_shape)

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def blur_model(input):
    blurred = tf.nn.depthwise_conv2d(input, blur_kernel_tensor, strides=[1,1,1,1], padding='SAME')
    return blurred

def blur_noise(input):
    blurred = tf.nn.depthwise_conv2d(input, blur_kernel_tensor, strides=[1,1,1,1], padding='SAME')
    return gaussian_noise_layer(blurred, std)

def blur_gramian(input):
    return blur_model(blur_model(input))


def identity(input):
    return input
