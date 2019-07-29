from preconditioned_neumann_network import PreconditionedNeumannNet
from src.operators_deblur_cifar import blur_gramian, blur_model, blur_noise
import png_utils_2d as png_utils
import os

def main():
    cwd = os.getcwd()

    # Point this to your training data.
    location_of_clean_data = cwd + '''/training_data/cifar_train/'''

    checkpoint_folder = cwd + '''/ckpts/'''
    checkpoint_filename = '''pnn_cifar_deblur.ckpt'''

    filestream = png_utils.PNG_Stream(location_of_clean_data)
    n_blocks = 6  # B in the Neumann networks paper
    image_dimension = 32  # Current version expects square images. This is easily modified.
    batch_size = 32
    n_samples = 30000  # Size of training dataset.
    starting_learning_rate = 1e-3  # Learning rate is decayed exponentially with a rate set inside the .train method.
    n_epochs = 100
    timelimit = 12240
    color_channels = 3

    learned_iterative_net = PreconditionedNeumannNet(forward_gramian=blur_gramian, corruption_model=blur_noise,
                                   forward_adjoint=blur_model,  iterations=n_blocks,
                                   image_dimension=image_dimension, batch_size=batch_size, color_channels=color_channels,
                                   n_training_samples=n_samples, initial_learning_rate=starting_learning_rate)
    learned_iterative_net.find_initial_conditions(checkpoint_folder, checkpoint_filename)
    learned_iterative_net.train(file_stream=filestream, n_epochs=n_epochs, timelimit=timelimit)

if __name__=="__main__":
    main()
