from __future__ import print_function

# Import MNIST data
import sys
import getopt
import input_data
import os.path
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

class Usage(Exception):
    def __init__ (self,msg):
        self.msg = msg

# Parameters
learning_rate =  1e-4
training_epochs = 200
batch_size = 100
display_step = 1

# Network Parameters
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

n_hidden_1 = 300# 1st layer number of features
n_hidden_2 = 100# 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# sets the threshold
prune_threshold_cov = 0.08
prune_threshold_fc = 1
# Frequency in terms of number of training iterations
prune_freq = 100
ENABLE_PRUNING = 0


def initialize_variables(weights_file_name):
    with open(weights_file_name,'rb') as f:
        wc1, wc2, wd1, out, bc1, bc2, bd1, bout = pickle.load(f)
        # weights, biases = pickle.load(f)
    weights_val = {
        'cov1': wc1,
        'cov2': wc2,
        'fc1': wd1,
        'fc2': out
    }
    biase_val = {
        'cov1': bc1,
        'cov2': bc2,
        'fc1': bd1,
        'fc2': bout
    }
    weights = weights_val
    biases = biase_val
    weights = {
        'cov1': tf.Variable(weights['cov1']),
        'cov2': tf.Variable(weights['cov2']),
        'fc1': tf.Variable(weights['fc1']),
        'fc2': tf.Variable(weights['fc2'])
    }
    biases = {
        'cov1': tf.Variable(biases['cov1']),
        'cov2': tf.Variable(biases['cov2']),
        'fc1': tf.Variable(biases['fc1']),
        'fc2': tf.Variable(biases['fc2'])
    }
    return (weights, biases)

def quantisation_4bits(matrix):
    q1 = tf.to_float(tf.logical_and(matrix <= 0.25, matrix > 0)) * 0.
    qn1 = tf.to_float(tf.logical_and(matrix >= -0.25, matrix <= 0)) * 0.
    q2 = tf.to_float(tf.logical_and(matrix <= 0.5, matrix > 0.25)) * 0.25
    qn2 = tf.to_float(tf.logical_and(matrix >= -0.5, matrix < -0.25)) * (-0.25)
    q3 = tf.to_float(tf.logical_and(matrix <= 0.75, matrix > 0.5)) * 0.5
    qn3 = tf.to_float(tf.logical_and(matrix >= -0.75, matrix < -0.5)) * (-0.5)
    q4 = tf.to_float(tf.logical_and(matrix <= 1, matrix > 0.75)) * 0.75
    qn4 = tf.to_float(tf.logical_and(matrix >= -1, matrix < -0.75)) * (-0.75)
    q5 = tf.to_float(tf.logical_and(matrix <= 1.25, matrix > 1)) * 1.0
    qn5 = tf.to_float(tf.logical_and(matrix >= -1.25, matrix < -1)) * (-1.0)
    q6 = tf.to_float(tf.logical_and(matrix <= 1.5, matrix > 1.25)) * 1.25
    qn6 = tf.to_float(tf.logical_and(matrix >= -1.5, matrix < -1.25)) * (-1.25)
    q7 = tf.to_float(tf.logical_and(matrix <= 1.75, matrix > 1.5)) * 1.5
    qn7 = tf.to_float(tf.logical_and(matrix >= -1.75, matrix < -1.5)) * (-1.5)
    q8 = tf.to_float(matrix > 1.75) * 1.75
    qn8 = tf.to_float(matrix < -1.75) * (-1.75)
    return (q1+qn1+q2+qn2+q3+qn3+q4+qn4+q5+qn5+q6+qn6+q7+qn7+q8+qn8)


def quantisation_4bits_np(matrix):
    q1 = np.to_float(np.logical_and(matrix <= 0.25, matrix > 0)) * 0.
    qn1 = np.to_float(np.logical_and(matrix >= -0.25, matrix <= 0)) * 0.
    q2 = np.to_float(np.logical_and(matrix <= 0.5, matrix > 0.25)) * 0.25
    qn2 = np.to_float(np.logical_and(matrix >= -0.5, matrix < -0.25)) * (-0.25)
    q3 = np.to_float(np.logical_and(matrix <= 0.75, matrix > 0.5)) * 0.5
    qn3 = np.to_float(np.logical_and(matrix >= -0.75, matrix < -0.5)) * (-0.5)
    q4 = np.to_float(np.logical_and(matrix <= 1, matrix > 0.75)) * 0.75
    qn4 = np.to_float(np.logical_and(matrix >= -1, matrix < -0.75)) * (-0.75)
    q5 = np.to_float(np.logical_and(matrix <= 1.25, matrix > 1)) * 1.0
    qn5 = np.to_float(np.logical_and(matrix >= -1.25, matrix < -1)) * (-1.0)
    q6 = np.to_float(np.logical_and(matrix <= 1.5, matrix > 1.25)) * 1.25
    qn6 = np.to_float(np.logical_and(matrix >= -1.5, matrix < -1.25)) * (-1.25)
    q7 = np.to_float(np.logical_and(matrix <= 1.75, matrix > 1.5)) * 1.5
    qn7 = np.to_float(np.logical_and(matrix >= -1.75, matrix < -1.5)) * (-1.5)
    q8 = np.to_float(matrix > 1.75) * 1.75
    qn8 = np.to_float(matrix < -1.75) * (-1.75)
    return (q1+qn1+q2+qn2+q3+qn3+q4+qn4+q5+qn5+q6+qn6+q7+qn7+q8+qn8)


def compute_weights_4bits(weights, biases):
    keys = ['cov1','cov2','fc1','fc2']
    # 4 bit. 1 bit sign, 1 bit integer, 2 bits fractional 00(0)-11(0.5+0.25)
    for key in keys:
        weights[key] = tf.floordiv(weights[key],0.125) * 0.125
        biases[key] = tf.floordiv(biases[key],0.125) * 0.125
    return (weights, biases)

def compute_weights_nbits(weights, biases, frac_bits):
    keys = ['cov1','cov2','fc1','fc2']
    # two defualt bits: 1 bit sign, 1 bit integer
    # 4 bit. 1 bit sign, 1 bit integer, 2 bits fractional 00(0)-11(0.5+0.25)
    interval = 0.5 / float(frac_bits)
    for key in keys:
        weights[key] = tf.floordiv(weights[key], interval) * interval
        biases[key] = tf.floordiv(biases[key], interval) * interval
    return (weights, biases)

def conv_network(x, weights, biases):
    conv = tf.nn.conv2d(x,
                        weights['cov1'],
                        strides = [1,1,1,1],
                        padding = 'VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['cov1']))
    pool = tf.nn.max_pool(
            relu,
            ksize = [1 ,2 ,2 ,1],
            strides = [1, 2, 2, 1],
            padding = 'VALID')

    conv = tf.nn.conv2d(pool,
                        weights['cov2'],
                        strides = [1,1,1,1],
                        padding = 'VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['cov2']))
    pool = tf.nn.max_pool(
            relu,
            ksize = [1 ,2 ,2 ,1],
            strides = [1, 2, 2, 1],
            padding = 'VALID')
    '''get pool shape'''
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [-1, pool_shape[1]*pool_shape[2]*pool_shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, weights['fc1']) + biases['fc1'])
    output = tf.matmul(hidden, weights['fc2']) + biases['fc2']
    return output , reshape

def calculate_non_zero_weights(weight):
    count = (weight != 0).sum()
    size = len(weight.flatten())
    return (count,size)

'''
Prune weights, weights that has absolute value lower than the
threshold is set to 0
'''

def prune_weights(threshold, weights, weight_mask):
    keys = ['cov1','cov2','fc1','fc2']
    for key in keys:
        weight_mask[key] = np.abs(weights[key].eval()) > threshold[key]
    with open('mask.pkl', 'wb') as f:
        pickle.dump(weight_mask, f)

def update_weights(grads_and_names):
    keys = ['cov1','cov2','fc1','fc2']
    for grad,var_name in grads_and_names:
        for key in keys:
            if (weights[key] == var_name):
                for i in range(1,9):
                    centroids_var[i] = centroids_var[i] - tf.reduces_sum((weights_index == i) * grad)

'''
mask gradients, for weights that are pruned, stop its backprop
'''
def mask_gradients(grads_and_names, weight_masks, weights, biases_masks, biases):
    new_grads = []
    keys = ['cov1','cov2','fc1','fc2']
    for grad, var_name in grads_and_names:
        # flag set if found a match
        flag = 0
        index = 0
        for key in keys:
            if (weights[key]== var_name):
                # print(key, weights[key].name, var_name)
                mask = weight_masks[key]
                new_grads.append((tf.multiply(tf.constant(mask, dtype = tf.float32),grad),var_name))
                flag = 1
            if (biases[key]== var_name):
                # print(key, weights[key].name, var_name)
                mask = biases_masks[key]
                new_grads.append((tf.multiply(tf.constant(mask, dtype = tf.float32),grad),var_name))
                flag = 1
        # if flag is not set
        if (flag == 0):
            new_grads.append((grad,var_name))
    return new_grads

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

'''
Define a training strategy
'''
def main(argv = None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts = argv
            for item, val in opts:
                if (item == '-quantisation_bits'):
                    q_bits = val
                if (item == '-parent_dir'):
                    parent_dir = val
                if (item == '-base_name'):
                    base_name = val
        except getopt.error, msg:
            raise Usage(msg)

        # obtain all weight masks
        mask_dir = parent_dir + 'masks/' + base_name + '.pkl'
        with open(mask_dir,'rb') as f:
            weights_mask,biases_mask = pickle.load(f)

        mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])
        keys = ['cov1','cov2','fc1','fc2']

        x_image = tf.reshape(x,[-1,28,28,1])

        weights_dir = parent_dir + 'weights/' + base_name + '.pkl'
        weights, biases = initialize_variables(weights_dir)
        weights,biases = compute_weights_nbits(weights, biases, q_bits)
        # Construct model
        pred, pool = conv_network(x_image, weights, biases)

        # Define loss and optimizer
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))

        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        org_grads = trainer.compute_gradients(cost)

        org_grads = [(ClipIfNotNone(grad), var) for grad, var in org_grads]
        new_grads = mask_gradients(org_grads, weights_mask, weights, biases_mask, biases)
        train_step = trainer.apply_gradients(new_grads)
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            keys = ['cov1','cov2','fc1','fc2']

            training_cnt = 0
            train_accuracy = 0
            accuracy_list = np.zeros(20)

            pre_train_acc = accuracy.eval({x:mnist.test.images, y: mnist.test.labels})
            print("Directly before pruning, Test Accuracy:", pre_train_acc)
            print(70*'-')
            print(weights['fc1'].eval())
            print(70*'-')
            print('Training starts ...')

            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    # execute a pruning
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    [_, c, train_accuracy] = sess.run([train_step, cost, accuracy], feed_dict = {
                            x: batch_x,
                            y: batch_y})
                    training_cnt = training_cnt + 1
                    accuracy_list = np.concatenate((np.array([train_accuracy]),accuracy_list[0:19]))
                    accuracy_mean = np.mean(accuracy_list)
                    if (i % 1000 == 0):
                        print('cost and acc:',c,accuracy_mean)
                    if (accuracy_mean > 0.99):
                        accuracy_list = np.zeros(20)
                        test_acc = accuracy.eval({  x:mnist.test.images,
                                                    y: mnist.test.labels})
                        print('Try quantize {} frac bits, test accuracy is {}'.format(q_bits, test_acc))
                        if (q_bits == 2):
                            threshold = 0.9
                        elif (q_bits == 4):
                            threshold = 0.99
                        elif (q_bits == 8):
                            threshold = 0.992
                        elif (q_bits == 16):
                            threshold = 0.993
                        elif (q_bits == 32):
                            threshold = 0.992

                        if (test_acc > threshold):
                            print('Training ends because accuracy is high')
                            with open(parent_dir+'weights/'+ 'quanfp' + str(q_bits) +'.pkl','wb') as f:
                                pickle.dump((
                                    weights['cov1'].eval(),
                                    weights['cov2'].eval(),
                                    weights['fc1'].eval(),
                                    weights['fc2'].eval(),
                                    biases['cov1'].eval(),
                                    biases['cov2'].eval(),
                                    biases['fc1'].eval(),
                                    biases['fc2'].eval(),
                                ),f)
                            print("saving model ...")
                            return (pre_train_acc, test_acc)
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

            print('Training ends because timeout, but still save the model')
            with open('weights/quanfp'+ str(q_bits) +'.pkl','wb') as f:
                pickle.dump((
                    weights['cov1'].eval(),
                    weights['cov2'].eval(),
                    weights['fc1'].eval(),
                    weights['fc2'].eval(),
                    biases['cov1'].eval(),
                    biases['cov2'].eval(),
                    biases['fc1'].eval(),
                    biases['fc2'].eval(),
                ),f)
            test_acc = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
            print("Test Accuracy:", test_acc)
            return (pre_train_acc, test_acc)

    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, "for help use --help"
        return 2

def weights_info(iter,  c, train_accuracy):
    print('This is the {}th iteration, cost is {}, accuracy is {}'.format(
        iter,
        c,
        train_accuracy
    ))

def prune_info(weights, counting):
    if (counting == 0):
        (non_zeros, total) = calculate_non_zero_weights(weights['cov1'].eval())
        print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['cov2'].eval())
        print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc2'].eval())
        print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    if (counting == 1):
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('take fc1 as example, {} nonzeros, in total {} weights'.format(non_zeros, total))

def mask_info(weights):
    (non_zeros, total) = calculate_non_zero_weights(weights['cov1'])
    print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    (non_zeros, total) = calculate_non_zero_weights(weights['cov2'])
    print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    (non_zeros, total) = calculate_non_zero_weights(weights['fc1'])
    print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    (non_zeros, total) = calculate_non_zero_weights(weights['fc2'])
    print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))

def write_numpy_to_file(data, file_name):
    # Write the array to disk
    with file(file_name, 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(data.shape))

        for data_slice in data:
            for data_slice_two in data_slice:
                np.savetxt(outfile, data_slice_two)
                outfile.write('# New slice\n')


if __name__ == '__main__':
    sys.exit(main())
