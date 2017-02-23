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
learning_rate = 0.001
training_epochs = 10
batch_size = 100
display_step = 1

# Network Parameters
Number_of_cluster= 8*2*2
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


with open('quantize_info.pkl','rb') as f:
    (weights_orgs, biases_orgs, cluster_index, centroids) = pickle.load(f)

centroids_var = {
    'cov1': tf.Variable(centroids['cov1']),
    'cov2': tf.Variable(centroids['cov2']),
    'fc1': tf.Variable(centroids['fc1']),
    'fc2': tf.Variable(centroids['fc2'])
}
# weights_index = {
#     'cov1': tf.Variable(cluster_index['cov1']),
#     'cov2': tf.Variable(cluster_index['cov2']),
#     'fc1': tf.Variable(cluster_index['fc1']),
#     'fc2': tf.Variable(cluster_index['fc2'])
# }
weights_index = {
    'cov1': tf.constant(cluster_index['cov1'],tf.float32),
    'cov2': tf.constant(cluster_index['cov2'],tf.float32),
    'fc1': tf.constant(cluster_index['fc1'],tf.float32),
    'fc2': tf.constant(cluster_index['fc2'],tf.float32)
}
biases = {
    'cov1': tf.Variable(biases_orgs['cov1']),
    'cov2': tf.Variable(biases_orgs['cov2']),
    'fc1': tf.Variable(biases_orgs['fc1']),
    'fc2': tf.Variable(biases_orgs['fc2'])
}

def compute_weights(weights_index, centroids_var):
    keys = ['cov1','cov2','fc1','fc2']
    weights = {}
    for key in keys:
        # weights[key] = tf.to_float(tf.equal(weights_index[key], 1)) * centroids_var[key][0]
        print(weights_index[key])
        for i in range(1, Number_of_cluster + 1):
            if (i == 1):
                weights[key] = tf.cast(tf.equal(weights_index[key], 1),tf.float32) * tf.cast(centroids_var[key][0], tf.float32)
            else:
                weights[key] = weights[key] + tf.to_float(tf.equal(weights_index[key], i)) * tf.cast(centroids_var[key][i-1], tf.float32)
    return weights


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

# def quantize_a_value(val):
#
#
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
def mask_gradients(grads_and_names, weight_masks, weights):
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

'''
plot weights and store the fig
'''
def plot_weights(weights,pruning_info):
        keys = ['cov1','cov2','fc1','fc2']
        fig, axrr = plt.subplots( 2, 2)  # create figure &  axis
        fig_pos = [(0,0), (0,1), (1,0), (1,1)]
        index = 0
        for key in keys:
            weight = weights[key].eval().flatten()
            # print (weight)
            size_weight = len(weight)
            weight = weight.reshape(-1,size_weight)[:,0:size_weight]
            x_pos, y_pos = fig_pos[index]
            #take out zeros
            weight = weight[weight != 0]
            # print (weight)
            hist,bins = np.histogram(weight, bins=100)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            axrr[x_pos, y_pos].bar(center, hist, align = 'center', width = width)
            axrr[x_pos, y_pos].set_title(key)
            index = index + 1
        fig.savefig('fig_v3/weights'+pruning_info)
        plt.close(fig)

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
            opts, args = getopt.getopt(argv[1:],'h')

        except getopt.error, msg:
            raise Usage(msg)

        # obtain all weight masks
        with open('mask.pkl','rb') as f:
            weights_mask = pickle.load(f)

        mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])
        keys = ['cov1','cov2','fc1','fc2']

        x_image = tf.reshape(x,[-1,28,28,1])
        # Construct model
        weights = compute_weights(weights_index, centroids_var)
        pred, pool = conv_network(x_image, weights, biases)

        # Define loss and optimizer
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        merged = tf.merge_all_summaries()
        saver = tf.train.Saver()

        org_grads = trainer.compute_gradients(cost)

        org_grads = [(ClipIfNotNone(grad), var) for grad, var in org_grads]
        new_grads = mask_gradients(org_grads, weights_mask, weights)
        # update_weights(new_grads)

        train_step = trainer.apply_gradients(new_grads)


        init = tf.initialize_all_variables()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            keys = ['cov1','cov2','fc1','fc2']

            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Test Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

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
