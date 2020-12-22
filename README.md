# neumann_networks_code
Neumann Networks for Inverse Problems in Imaging

# This repository is old and busted

Please refer to the PyTorch implementation here: https://github.com/dgilton/iterative_reconstruction_networks
There appear to be some errors in this code - while some folks can get the networks to train, others have experienced issues. The PyTorch code is much more thoroughly tested and will be more wide-ranging. All the same I'm leaving this code here to remind future generations not to code like I did two years ago.

## Neumann Networks

This is the repository for code related to *Neumann Networks for Inverse Problems in Imaging*.
The paper is available on the Arxiv at: [https://arxiv.org/abs/1901.03707](https://arxiv.org/abs/1901.03707)

A brief introduction is available on the [project website](https://dgilton.github.io/neumann_networks/).
This README contains instructions for implementation and code documentation.

## Dependencies
This code has been tested on the following:

Tensorflow: 1.13.1
Python: 3.7.3 (But most modern 3.6 should be fine too)
Cuda: 9.0
Auxiliary utilities use Scipy 0.9.0 and require PIL. If you have a more recent version of
Scipy you'll have to replace some scipy.misc.imread() calls to imageio.imread() calls.

If you experience issues, please contact the corresponding author via email or submit an issue.
Be especially careful that your Tensorflow version is the same as listed.
I highly recommend installing Tensorflow via conda in a virtualenv.

## Using the code

Training and testing are done through driver scripts, written in Python. These scripts
may be run as simply as
```
python my_driver_script.py >> ./local_log_file.txt
```
Example driver scripts for training and testing are provided by default in the top
level of this repo.

#### Training data
Training data is assumed to be located in the ./training_data/ directory. During our 
training process, each dataset had its own subdirectory, so CelebA training data would be 
located in ./training_data/celeba_train/ for example.  

The .train method of the methods implemented here assumes that data is accessed through a 
class, I refer to as datastreams.
All that is required in the training procedure is that these classes contain a next_batch()
method, which takes as a single input the batch size, and returns a numpy array of size 
[batch_size, image_dimension1, image_dimension2, number_of_channels]. I recommend running any
preprocessing steps in this next_batch() method.

If you would like to restart training from a checkpoint, just run the find_initial_conditions() 
method, pointed at the checkpoint you want to start from. Training will initialize using that
method. Be careful with the training schedule, since the checkpoints by default include the current 
state of training, so the decay schedule may make your gradient steps smaller than desired.

#### Forward Models
The NeumannNet() class and other iterative methods here ask for three separate functions as 
input: forward_gramian, corruption_model, and forward_adjoint. These must be functions which 
operate on Tensorflow tensors, and implement certain fixed parts of the iterative models.

The forward_gramian implements X^T (X ( \beta ) ), the gramian of the forward operator. The 
forward_adjoint model implements X^T (\beta), the adjoint of the forward operator. The 
corruption_model function takes in \beta and returns X( \beta ) + \epsilon. Only the corruption_model 
function should add noise.

For example, in a superresolution setting, these functions are simple to implement. 
corruption_model() would correspond to downsampling a full-size image and adding some small amount
of noise. The forward_adjoint() function would directly upsample a downsampled image, but not add any noise.
The forward_gramian() function would downsample a full-size image, and then upsample the image again.
See the source files that have the prefix operators_* for more examples.

#### Testing
Testing is also done by driver scripts, but requires a pretrained checkpoint. Make sure to run
the find_initial_conditions() method before testing.

An example testing driver script is provided - on my computer it should be as simple as running 
python cifar_deblur_testing_driver.py . Checkpoints are provided for deblur+noise for the 
preconditioned Neumann network, the Neumann Network, and Gradient Descent network. Note that 
the learned component here is different than in the original paper - we have found the new architecture
improves results both in terms of PSNR and speed.

#### Custom preconditioners
If you would like to use your own preconditioner, just import it in the header of the 
preconditioned_neumann_network.py file.

If you have questions, requests, ideas, or are interested in a resource that is not provided
here or on the Github (like a pretrained checkpoint for a problem or method) please contact
Davis Gilton via email. His email is his last name at wisc.edu .
