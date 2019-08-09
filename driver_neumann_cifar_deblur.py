from src.neumann_network import NeumannNet
from src.operators_deblur_cifar import blur_gramian, blur_model, blur_noise
import src.png_utils_2d as png_utils
import os

def main():
    cwd = os.getcwd()

    # Point this to your training data.
    location_of_clean_data = cwd + '''/training_data/cifar_train/'''

    checkpoint_folder = cwd + '''/ckpts/'''
    checkpoint_filename = '''neumann_cifar_deblur.ckpt'''

    # filestream must implement a method called next_batch() which takes in a batch_size parameter and returns
    # a numpy array of size [batch_size, image_dimension, image_dimension, color_channels]. All preprocessing happens
    # inside this function, so if you'd like to do preprocessing in tensorflow you'll have to modify the graph in
    # the NeumannNet constructor.
    filestream = png_utils.PNG_Stream_randomorder(location_of_clean_data)
    n_blocks = 6 # B in the Neumann networks paper
    image_dimension = 32 # Current version expects square images. This is easily modified.
    batch_size = 32
    n_samples = 30000 # Size of training dataset.
    starting_learning_rate = 1e-3 # Learning rate is decayed exponentially with a rate set inside the .train method.
    n_epochs = 100
    color_channels = 3 # Number of spectral channels. MRI should use 2, remote sensing may have more.

    # Our server's scheduler kills all jobs after a certain time period, so we stop early to clean up and restart.
    # If you don't want this behavior, set timelimit to 0.
    timelimit = 12240


    # forward_gramian, corruption_model, and forward_adjoint need to be tensorflow functions. forward_adjoint should
    # implement $X^\T ()$, forward_gramian should implement $X^\T X ()$, and corruption_model should be
    # $X() + \epsilon$. If you're going to add noise, do it in corruption_model.
    learned_iterative_net = NeumannNet(forward_gramian=blur_gramian, corruption_model=blur_noise,
                                   forward_adjoint=blur_model,  iterations=n_blocks,
                                   image_dimension=image_dimension, batch_size=batch_size, color_channels=color_channels,
                                   n_training_samples=n_samples, initial_learning_rate=starting_learning_rate)
    # Finds any
    learned_iterative_net.find_initial_conditions(checkpoint_folder, checkpoint_filename)
    learned_iterative_net.train(file_stream=filestream, n_epochs=n_epochs, timelimit=timelimit)

if __name__=="__main__":
    main()
