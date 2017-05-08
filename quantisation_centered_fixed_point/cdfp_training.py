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

def compute_weights_nbits(weights, biases, frac_bits, dynamic_range, c_pos, c_neg, central_value):
    keys = ['cov1','cov2','fc1','fc2']
    # two defualt bits: 1 bit sign, 1 bit integer
    frac_range = 2 ** frac_bits - 1
    max_range = 0.5 ** (frac_bits) * frac_range
    interval =  0.5 ** (frac_bits)
    weights_new = {}
    biases_new = {}
    for key in keys:
        upper_part_pos = weights[key] > central_value[key]
        upper_part_pos = tf.cast(upper_part_pos, dtype = tf.float32)
        lower_part_pos = weights[key] <= central_value[key]
        lower_part_pos = tf.cast(lower_part_pos, dtype = tf.float32)
        for i in range(dynamic_range):
            if (i == 0):
                next_max_range = (0.5 ** (frac_bits)) * frac_range * (0.5 ** (i+1))

                weight_regulate = upper_part_pos * (weights[key] - c_pos[key]) + lower_part_pos * (weights[key] - c_neg[key])

                w_pos = tf.cast(tf.abs(weight_regulate) > next_max_range, dtype=tf.float32)
                b_pos = tf.cast(tf.abs(biases[key]) > next_max_range, dtype=tf.float32)
                w_val = weights[key] * w_pos
                b_val = biases[key] * b_pos
                tmp =  w_pos * (c_pos * upper_part_pos + c_neg * lower_part_pos)
                weights_new[key] = tf.floordiv( w_val, interval) * interval
                biases_new[key] = tf.floordiv( b_val, interval) * interval
            elif (i == dynamic_range - 1):
                interval_dr = 0.5 ** (frac_bits + i)
                max_range = (0.5 ** (frac_bits)) * frac_range * (0.5 ** (i))
                weight_regulate = upper_part_pos * (weights[key] - c_pos[key]) + lower_part_pos * (weights[key] - c_neg[key])

                w_pos =tf.abs(weight_regulate) <= max_range
                b_pos =tf.abs(biases[key]) <= max_range
                w_pos = tf.cast(w_pos, dtype=tf.float32)
                b_pos = tf.cast(b_pos, dtype=tf.float32)
                w_val = weights[key] * w_pos
                b_val = biases[key] * b_pos
                weights_new[key] += tf.floordiv( w_val, interval) * interval_dr  + w_pos * (c_pos * upper_part_pos + c_neg * lower_part_pos)
                biases_new[key] += tf.floordiv(b_val, interval_dr) * interval_dr
            else:
                interval_dr = 0.5 ** (frac_bits + i)
                max_range = (0.5 ** (frac_bits)) * frac_range * (0.5 ** (i))
                next_max_range = (0.5 ** (frac_bits)) * frac_range * (0.5 ** (i+1))
                weight_regulate = upper_part_pos * (weights[key] - c_pos[key]) + lower_part_pos * (weights[key] - c_neg[key])

                w_pos = tf.logical_and((tf.abs(weight_regulate) <= (max_range)), (tf.abs(weight_regulate) > next_max_range))
                b_pos = tf.logical_and((tf.abs(biases[key]) <= (max_range)), (tf.abs(biases[key]) > next_max_range))
                w_pos = tf.cast(w_pos, dtype=tf.float32)
                b_pos = tf.cast(b_pos, dtype=tf.float32)
                w_val = weights[key] * w_pos
                b_val = biases[key] * b_pos
                weights_new[key] += tf.floordiv( w_val, interval) * interval_dr  + w_pos * (c_pos * upper_part_pos + c_neg * lower_part_pos)
                biases_new[key] += tf.floordiv(b_val, interval_dr) * interval_dr
    return (weights_new, biases_new)

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
                if (item == '-d_range'):
                    dynamic_range = val
                if (item == '-parent_dir'):
                    parent_dir = val
                if (item == '-base_name'):
                    base_name = val
                if (item == '-c_pos'):
                    c_pos = val
                if (item == '-c_neg'):
                    c_neg = val
                if (item == '-central_value'):
                    central_value = val
        except getopt.error, msg:
            raise Usage(msg)

        # obtain all weight masks
        mask_dir = parent_dir + 'masks/' + base_name + '.pkl'
        with open(mask_dir,'rb') as f:
            weights_mask = pickle.load(f)
        biases_mask = {
            'cov1': np.ones([20]),
            'cov2': np.ones([50]),
            'fc1': np.ones([500]),
            'fc2': np.ones([10])
        }

        mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])
        keys = ['cov1','cov2','fc1','fc2']

        x_image = tf.reshape(x,[-1,28,28,1])

        weights_dir = parent_dir + 'weights/' + base_name + '.pkl'
        weights, biases = initialize_variables(weights_dir)
        new_weights, new_biases = compute_weights_nbits(weights, biases, q_bits, dynamic_range, c_pos, c_neg, central_value)
        # Construct model
        pred, pool = conv_network(x_image, new_weights, new_biases)

        # Define loss and optimizer
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))

        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        org_grads = trainer.compute_gradients(cost)

        org_grads = [(ClipIfNotNone(grad), var) for grad, var in org_grads]
        new_grads = mask_gradients(org_grads, weights_mask, new_weights, biases_mask, new_biases)
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
                        test_acc = accuracy.eval({  x:mnist.test.images,
                                                    y: mnist.test.labels})
                        print('Try quantize {} frac bits, test accuracy is {}'.format(q_bits, test_acc))
                        if (test_acc > 0.9936):
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
