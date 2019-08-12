# from src.gradient_descent_network import GradientDescentNet as Net
from src.preconditioned_neumann_network import PreconditionedNeumannNet as Net
# from src.neumann_network import NeumannNet as Net

from src.operators_blur_cifar import blur_gramian, blur_model, blur_noise
import src.png_utils_2d as png_utils
import os
import numpy as np
from scipy import misc

def main():
    cwd = os.getcwd()

    # Point this to your testing data.
    location_of_test_data = cwd + '''/testing_data/cifar/'''

    checkpoint_folder = cwd + '''/ckpts/'''
    checkpoint_filename = '''pnn_cifar_deblur_2.ckpt'''
    output_folder = '''/images/cifar/'''

    if not os.path.exists(cwd + output_folder):
        os.mkdir(cwd + output_folder)

    # filestream must implement a method called next_batch() which takes in a batch_size parameter and returns
    # a numpy array of size [batch_size, image_dimension, image_dimension, color_channels]. All preprocessing happens
    # inside this function, so if you'd like to do preprocessing in tensorflow you'll have to modify the graph in
    # the NeumannNet constructor.
    filestream = png_utils.PNG_Stream_ordered1(location_of_test_data)
    n_blocks = 6 # B in the Neumann networks paper
    image_dimension = 32 # Current version expects square images. This is easily modified.
    batch_size = 32
    n_samples = 30000 # Size of training dataset.
    starting_learning_rate = 1e-3 # Learning rate is decayed exponentially with a rate set inside the .train method.
    color_channels = 3 # Number of spectral channels. MRI should use 2, remote sensing may have more.


    # forward_gramian, corruption_model, and forward_adjoint need to be tensorflow functions. forward_adjoint should
    # implement $X^\T ()$, forward_gramian should implement $X^\T X ()$, and corruption_model should be
    # $X() + \epsilon$. If you're going to add noise, do it in corruption_model.
    learned_iterative_net = Net(forward_gramian=blur_gramian, corruption_model=blur_noise,
                                   forward_adjoint=blur_model,  iterations=n_blocks,
                                   image_dimension=image_dimension, batch_size=batch_size, color_channels=color_channels,
                                   n_training_samples=n_samples, initial_learning_rate=starting_learning_rate)
    # Finds any
    learned_iterative_net.find_initial_conditions(checkpoint_folder, checkpoint_filename)
    reconstructed_imgs, true_imgs = learned_iterative_net.reconstruct_procedure(file_stream=filestream, n_batches=8)

    n_test_images = np.shape(true_imgs)[0]
    size = image_dimension * image_dimension * color_channels
    mse = []
    for ii in range(n_test_images):
        mse.append(-10 * np.log10(
            np.sum(np.square(reconstructed_imgs[ii, :, :, :] - true_imgs[ii, :, :, :]), axis=(0, 1, 2)) / size))
        # print(np.max(reconstructed_imgs[ii,:,:,:][:]))
        misc.toimage(reconstructed_imgs[ii, :, :, :], cmin=0.0, cmax=np.max(reconstructed_imgs[ii,:,:,:][:])).save(
            cwd + output_folder + str(ii) + ".png")
    print(np.median(mse))

if __name__=="__main__":
    main()
