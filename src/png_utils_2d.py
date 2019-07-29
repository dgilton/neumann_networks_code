from PIL import Image
import numpy as np
import random, os

import re


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def directory_filelist(target_directory):
    file_list = [f for f in os.listdir(target_directory)
                 if os.path.isfile(os.path.join(target_directory, f))]
    file_list = list(file_list)
    # print(len(file_list))
    try:
        file_list.remove('.DS_Store')
    except ValueError:
        pass
    file_list = [f for f in file_list if not f.startswith('.')]
    return file_list

def load_png(file_name):
    with open(file_name,'rb') as f:
        img = Image.open(f).convert("RGB")
        image = np.asarray(img)
    return image


def fetch_batch_of_tiff_layers(list_of_images, batch_size, layer):
    batch_list = random.sample(list_of_images, batch_size)
    array_of_image_vectors = np.asarray([load_png(f, layer) for f in batch_list])
    return array_of_image_vectors

class PNG_Stream():
    def __init__(self, target_directory):
        filelist = directory_filelist(target_directory)
        self.full_filelist = [target_directory + single_file for single_file in filelist]

    def next_batch(self, batch_size):
        batch_list = random.sample(self.full_filelist, batch_size)
        array_of_images = np.asarray([load_png(f) for f in batch_list])
        return array_of_images / 255.0

class Singleimage_PNG_Stream():
    def __init__(self, target_noisy_directory, target_clean_directory):
        # self.noisy_full_filelist = [target_noisy_directory + single_file for single_file in filelist]
        # self.clean_full_filelist = [target_clean_directory + single_file for single_file in filelist]
        self.noisy_directory = target_noisy_directory
        self.clean_directory = target_clean_directory

    def next_batch(self, batch_size):
        batch_list = ['89.png']
        array_of_noisy_images = np.asarray([load_png(self.noisy_directory + f) for f in batch_list])
       	array_of_clean_images = np.asarray([load_png(self.clean_directory + f) for f in batch_list])
        return array_of_noisy_images / 255.0, array_of_clean_images / 255.0

class Coupled_PNG_Stream():
    def __init__(self, target_noisy_directory, target_clean_directory):
        self.filelist = directory_filelist(target_noisy_directory)
        # self.noisy_full_filelist = [target_noisy_directory + single_file for single_file in filelist]
        # self.clean_full_filelist = [target_clean_directory + single_file for single_file in filelist]
        self.noisy_directory = target_noisy_directory
        self.clean_directory = target_clean_directory

    def next_batch(self, batch_size):
        batch_list = random.sample(self.filelist, batch_size)
        array_of_noisy_images = np.asarray([load_png(self.noisy_directory + f) for f in batch_list])
        array_of_clean_images = np.asarray([load_png(self.clean_directory + f) for f in batch_list])
        return array_of_noisy_images / 255.0, array_of_clean_images / 255.0

class Coupled_PNG_Stream_subset():
    def __init__(self, target_noisy_directory, target_clean_directory, subset_size, seed):
        init_filelist = directory_filelist(target_noisy_directory)
        random.seed(seed)
        self.filelist = random.sample(init_filelist, subset_size)
        print(self.filelist)
        # self.noisy_full_filelist = [target_noisy_directory + single_file for single_file in filelist]
        # self.clean_full_filelist = [target_clean_directory + single_file for single_file in filelist]
        self.noisy_directory = target_noisy_directory
        self.clean_directory = target_clean_directory

    def next_batch(self, batch_size):
        batch_list = random.sample(self.filelist, batch_size)
        array_of_noisy_images = np.asarray([load_png(self.noisy_directory + f) for f in batch_list])
        array_of_clean_images = np.asarray([load_png(self.clean_directory + f) for f in batch_list])
        return array_of_noisy_images / 255.0, array_of_clean_images / 255.0

class PNG_Stream_ordered():
    def __init__(self, target_directory):
        filelist = directory_filelist(target_directory)
        self.full_filelist = [target_directory + single_file for single_file in filelist]
        self.list_length = len(self.full_filelist) + 1
        self.counter = 0

    def next_batch(self, batch_size):
        batch_list = self.full_filelist[(self.counter%self.list_length):((self.counter+batch_size)%self.list_length)]
        # print(batch_list)
        self.counter += batch_size
        array_of_images = np.asarray([load_png(f) for f in batch_list])
        return array_of_images

class PNG_Stream_ordered1():
    def __init__(self, target_directory):
        filelist = directory_filelist(target_directory)
        self.full_filelist = [target_directory + single_file for single_file in filelist]
        self.list_length = len(self.full_filelist) + 1
        self.counter = 0

    def next_batch(self, batch_size):
        batch_list = sorted_nicely(self.full_filelist[(self.counter%self.list_length):((self.counter+batch_size)%self.list_length)])
        # print(batch_list)
        self.counter += batch_size
        array_of_images = np.asarray([load_png(f) for f in batch_list])
        return array_of_images / 255.0
