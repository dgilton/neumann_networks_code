import numpy as np
import tensorflow as tf
from PIL import Image
import os, pickle, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def list_files_in_directory(target_directory):
    file_list = [f for f in os.listdir(target_directory)
                 if os.path.isfile(os.path.join(target_directory, f))]
    # Because I often work on a Mac, I attempt to remove the .DS_Store file,
    # which really does no good for most practical purposes.
    try:
        file_list.remove('.DS_Store')
    except ValueError:
        print('No DS_Store to remove')
    return file_list


def generate_patches(image, patch_h, patch_w):
    '''Splits an image into patches of size patch_h x patch_w
   Input: image of shape [image_h, image_w, image_ch]
   Output: batch of patches shape [n, patch_h, patch_w, image_ch]
   '''

    pad = [[0, 0], [0, 0]]
    image_ch = 3
    p_area = patch_h * patch_w

    patches = tf.space_to_batch_nd(image, [patch_h, patch_w], pad)
    patches = tf.split(patches, p_area, 0)
    patches = tf.stack(patches, 3)
    patches = tf.reshape(patches, [-1, patch_h, patch_w, image_ch])

    return patches


def reconstruct_image(patches, image_h, image_w, n_images):
    '''Reconstructs an image from patches of size patch_h x patch_w
   Input: batch of patches shape [n, patch_h, patch_w, patch_ch]
   Output: image of shape [image_h, image_w, patch_ch]
   '''

    pad = [[0, 0], [0, 0]]
    patch_h = patches.shape[1].value
    patch_w = patches.shape[2].value
    patch_ch = patches.shape[3].value
    p_area = patch_h * patch_w
    h_ratio = image_h // patch_h
    w_ratio = image_w // patch_w

    image = tf.reshape(patches, [n_images, h_ratio, w_ratio, p_area, patch_ch])
    image = tf.split(image, p_area, 3)
    # print(image.shape)
    image = tf.stack(image, 0)
    image_storage = tf.zeros(shape=[0, image_h, image_w, patch_ch])
    for ii in range(n_images):
        reshaped_img = tf.reshape(image[:,ii,:,:,:,:], [p_area, h_ratio, w_ratio, patch_ch])
        reshaped_img = tf.batch_to_space_nd(reshaped_img, [patch_h, patch_w], pad)
        image_storage = tf.concat([image_storage, reshaped_img], axis=0)

    return image_storage

# Sometimes you just want to unpack a pcl file into a csv
def pickle2csv(in_name, out_name):
    with open(in_name, 'rb') as pickle_file:
        numpy_array = pickle.load(pickle_file)

    np.savetxt(out_name, numpy_array)
    return 0


def save_1d_line_plot_as_png(x, y, title='title', saved_name='myplot'):
    fig = plt.figure()
    plt.plot(x, y)
    plt.title(title)
    fig.savefig(saved_name + ".png")
    plt.clf()

# Xavier Initialization
def xavier_init(fan_in, fan_out, constant=1):
    """Initialize network weights with Xavier Initialization"""
    low = -constant*np.sqrt(6/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

def save_matrix_as_png(matrix, title='title', saved_name='mymatrix'):
    fig = plt.figure()
    # 

def save_matrix_as_csv(matrix, output_name):
    np.savetxt(output_name, matrix, delimiter=',')


class MNIST(object):
    def __init__(self):
        mnist = tf.contrib.learn.datasets.mnist.read_data_sets("fashion_MNIST-data")
        self.train_data = mnist.train.images
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        self.eval_data = mnist.test.images
        self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    def next_batch(self, batch_size):
        train_indices = np.random.randint(self.train_data.shape[0], size=batch_size)
        return self.train_data[train_indices]
        # return self.train_data[train_indices, :], self.train_labels[train_indices]

    def test_batch(self, batch_size):
        eval_indices = np.random.randint(self.eval_data.shape[0], size=batch_size)
        return self.eval_data[eval_indices, :], self.eval_labels[eval_indices]
