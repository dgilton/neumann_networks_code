import tensorflow as tf
import numpy as np
import scipy.linalg as scl

# Any preconditioning or pseudoinverse operator may be used. We use
# cg iterations to calculate an approximate pseudoinverse. Code is based on
# a similar implementation by MoDL.
from MoDL_utils import cg_pseudoinverse

from learned_component_resnet_nblock import nblock_resnet
import os, sys, time

def stdprint(input):
    sys.stdout.write(input+"\n")
    sys.stdout.flush()

######################################################
global_step = tf.Variable(0)
try:
    NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
except KeyError:
    stdprint("OMP_NUM_THREADS does not exist.")
    NUM_THREADS = 4 # The minimum cores on any cpu at ttic or in lab

stdprint("Number of threads available: " + str(NUM_THREADS))

######################################################
class PreconditionedNeumannNet(object):
    def __init__(self, forward_gramian, corruption_model, forward_adjoint, iterations, image_dimension, batch_size,
                 color_channels, n_training_samples, initial_learning_rate):
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF. This will be slower than normal.")
        self.num_iters = iterations
        self.image_dimension = image_dimension
        self.image_size = image_dimension * image_dimension # It thinks the images are square
        self.batch_size = batch_size
        self.n_batches_in_dataset = int(n_training_samples / self.batch_size)
        self.initial_learning_rate = initial_learning_rate

        self.true_beta = tf.placeholder(tf.float32, shape=[self.batch_size, self.image_dimension, self.image_dimension, color_channels])

        self.resnet = nblock_resnet()
        self.checkpoint_name = ""
        self.has_initialized = False
        self.eta = tf.get_variable(name='eta', initializer=0.1, dtype=tf.float32, trainable=True)

        network_input = forward_adjoint(corruption_model(self.true_beta))
        network_input = cg_pseudoinverse(forward_gramian=forward_gramian, rhs=network_input, eta=self.eta)
        runner = network_input
        neumann_sum = runner

        ############################################################
        # Build the network. Don't do anything else #
        ############################################################
        for ii in range(iterations):
            linear_component = self.eta * cg_pseudoinverse(forward_gramian=forward_gramian, rhs=runner, eta=self.eta)
            if tf.test.gpu_device_name():
                with tf.device('/gpu:1'):
                    regularizer_output = self.resnet.network(input=runner, is_training=True,
                                                             n_residual_blocks=1)
            else:
                regularizer_output = self.resnet.network(input=runner, is_training=True,
                                                         n_residual_blocks=1)
            learned_component = -(regularizer_output + runner)
            runner = linear_component + learned_component
            neumann_sum = neumann_sum + runner

        self.output = neumann_sum

    def find_initial_conditions(self, checkpoint_location, checkpoint_filename):
        self.checkpoint_name = checkpoint_location + checkpoint_filename
        for fname in os.listdir(checkpoint_location):
            if fname.startswith(checkpoint_filename):
                self.has_initialized = True
                break


    def train(self, file_stream, n_epochs, timelimit=0):
        data_set = file_stream
        beta_hat = self.output
        training_rounds = int(self.n_batches_in_dataset * n_epochs)

        if timelimit is not 0:
            start_time = time.time()

        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,
                                              inter_op_parallelism_threads=NUM_THREADS,
                                              allow_soft_placement=True)) as sess:
            network_saver = tf.train.Saver(tf.trainable_variables())

            learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step, 500, decay_rate=0.97)

            recon_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.true_beta - beta_hat), axis=[1, 2, 3])))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                gradients, variables = zip(*optimizer.compute_gradients(recon_loss))
                # gradients, _ = tf.clip_by_global_norm(gradients, 2.0)
                train_step = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
            stdprint("Initializing")
            tf.global_variables_initializer().run()
            if self.has_initialized:
                network_saver.restore(sess, self.checkpoint_name)

            for ii in range(training_rounds):
                #stdprint("Getting data")
                true_beta_img = data_set.next_batch(self.batch_size)
                #stdprint("backprop")
                _, current_loss, global_step_number = sess.run([train_step, recon_loss, global_step],
                                                  feed_dict={self.true_beta: true_beta_img})
                if global_step_number % 10 == 0:
                    stdprint("Iteration: " + str(global_step_number) + ", Loss: " + str(current_loss))
                if global_step_number % 100 == 0:
                    network_saver.save(sess, self.checkpoint_name)
                if timelimit is not 0:
                    current_time = time.time()
                    if timelimit < (current_time - start_time):
                        network_saver.save(sess, self.checkpoint_name)
                        return

            network_saver.save(sess, self.checkpoint_name)

    def reconstruct_procedure(self, file_stream, n_batches):
        if not self.has_initialized:
            stdprint('Initialize before trying to use the network for reconstruction.')
            exit()

        network_saver = tf.train.Saver(tf.trainable_variables())

        resulting_img_storage = np.zeros(shape=(n_batches*self.batch_size, self.image_dimension, self.image_dimension, 3))
        test_img_storage = np.zeros(
            shape=(n_batches * self.batch_size, self.image_dimension, self.image_dimension, 3))

        with tf.Session() as sess:
            network_saver.restore(sess, self.checkpoint_name)
            for ii in range(n_batches):
                test_input = file_stream.next_batch(self.batch_size)
                resulting_img = sess.run(self.output, feed_dict={self.true_beta: test_input})
                start_index = ii * self.batch_size
                end_index = (ii+1) * self.batch_size
                test_img_storage[start_index:end_index, :,:,:] = test_input
                resulting_img_storage[start_index:end_index, :,:,:] = resulting_img

        return resulting_img_storage, test_img_storage


def main():
    print('This must be run from an outside driver script.')


if __name__=="__main__":
    main()
