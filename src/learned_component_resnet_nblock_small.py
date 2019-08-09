from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.layers.normalization import BatchNormalization

def batch_norm(net_input, is_training, trainable=True, momentum=0.99,
           name="BatchNorm"):
    layer = tf.keras.layers.BatchNormalization(
        epsilon=1E-5, center=True, scale=True, momentum=momentum,
        trainable=trainable, name=name)
    add_elements_to_collection(layer.updates, tf.GraphKeys.UPDATE_OPS)

    return layer.apply(net_input, training=is_training)

def add_elements_to_collection(elements, collection_list):
    elements = tf.contrib.framework.nest.flatten(elements)
    collection_list = tf.contrib.framework.nest.flatten(collection_list)
    for name in collection_list:
       collection = tf.get_collection_ref(name)
       collection_set = set(collection)
       for element in elements:
           if element not in collection_set:
                collection.append(element)

def leaky_relu(x):
    return tf.maximum(0.1*x,x)

class nblock_resnet(object):
  
    def __init__(self):
        self.activation_fn = leaky_relu
        #self.activation_fn = tf.nn.relu

    def residual_block(self, input, is_training, n_channels, block_number):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        with tf.variable_scope('residual_block_'+str(block_number), reuse=tf.AUTO_REUSE):
            ### Convolutional layer 1
            conv_kernel_1 = tf.get_variable(name='kernel_1',initializer=initializer, shape=[3,3,n_channels, n_channels])
            conv_bias_1 = tf.get_variable(name='bias_1', initializer=0.001)
            conv_layer_1 = tf.nn.conv2d(input=input, filter=conv_kernel_1, strides=[1,1,1,1], padding='SAME') + conv_bias_1
            conv_layer_1 = batch_norm(conv_layer_1, is_training=is_training, name='BatchNorm1')
            conv_layer_1 = self.activation_fn(conv_layer_1)

            ### Convolutional layer 2
            conv_kernel_2 = tf.get_variable(name='kernel_2', initializer=initializer, shape=[3, 3, n_channels, n_channels])
            conv_bias_2 = tf.get_variable(name='bias_2', initializer=0.001)
            conv_layer_2 = tf.nn.conv2d(input=conv_layer_1, filter=conv_kernel_2, strides=[1, 1, 1, 1], padding='SAME') + conv_bias_2
            conv_layer_2 = batch_norm(conv_layer_2, is_training=is_training, name='BatchNorm2')
            conv_layer_2 = self.activation_fn(conv_layer_2)

            return input + conv_layer_2


    def network(self, input, is_training, n_residual_blocks):
        n_intermediate_channels = 128
        initializer = tf.contrib.layers.xavier_initializer_conv2d()

        with tf.variable_scope('learned_component', reuse=tf.AUTO_REUSE):
            patch_means = tf.reduce_mean(input, axis=(1,2), keep_dims=True)
            input = input - patch_means

            dimension_fit_kernel = tf.get_variable(name='dimension_fit', initializer=initializer, shape=[1,1,3, n_intermediate_channels])
            conv_bias_initial = tf.get_variable(name='bias_initial', initializer=0.001)
            conv_layer_initial = tf.nn.conv2d(input=input, filter=dimension_fit_kernel, strides=[1, 1, 1, 1],
                                    padding='SAME') + conv_bias_initial

            residual_output = conv_layer_initial

            for ii in range(n_residual_blocks):
                residual_output = self.residual_block(residual_output, is_training=is_training, n_channels=n_intermediate_channels, block_number=ii)

            ### 1x1 convolutions
            conv_kernel_0 = tf.get_variable(name='kernel_0', initializer=initializer, shape=[1, 1, n_intermediate_channels, n_intermediate_channels])
            conv_bias_0 = tf.get_variable(name='bias_0', initializer=0.001)
            conv_layer_0 = tf.nn.conv2d(input=residual_output, filter=conv_kernel_0, strides=[1, 1, 1, 1],
                                    padding='SAME') + conv_bias_0
            conv_layer_0 = self.activation_fn(conv_layer_0)
            conv_kernel_1 = tf.get_variable(name='kernel_1', initializer=initializer, shape=[1, 1, n_intermediate_channels, n_intermediate_channels])
            conv_bias_1 = tf.get_variable(name='bias_1', initializer=0.001)
            conv_layer_1 = tf.nn.conv2d(input=conv_layer_0, filter=conv_kernel_1, strides=[1, 1, 1, 1],
                                    padding='SAME') + conv_bias_1
            conv_layer_1 = self.activation_fn(conv_layer_1)

            conv_kernel_2 = tf.get_variable(name='kernel_2', initializer=initializer, shape=[1, 1, n_intermediate_channels, 3])
            conv_bias_2 = tf.get_variable(name='bias_2', initializer=0.001)
            conv_layer_2 = tf.nn.conv2d(input=conv_layer_1, filter=conv_kernel_2, strides=[1, 1, 1, 1],
                                    padding='SAME') + conv_bias_2

            conv_layer_2 = conv_layer_2 + patch_means
            return conv_layer_2

  

