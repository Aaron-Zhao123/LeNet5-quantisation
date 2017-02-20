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
import sklearn
from sklearn.cluster import KMeans


class Usage(Exception):
    def __init__ (self,msg):
        self.msg = msg

# Parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 100
display_step = 1
Number_of_cluster = 4

# Network Parameters
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

n_hidden_1 = 300# 1st layer number of features
n_hidden_2 = 100# 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
INITAL = 0
# FILE_NAME = "pcov96pfc96.pkl"
# FILE_NAME = '/Users/aaron/Projects/Mphil_project/tmp_asyn_prune/pcov91pcov91pfc995pfc91.pkl'
FILE_NAME = '/home/ubuntu/LENet5-431K/tmp/pcov91pcov91pfc995pfc91.pkl'
pruning_number = 10
if (INITAL == 0):
    with open(FILE_NAME,'rb') as f:
        wc1, wc2, wd1, out, bc1, bc2, bd1, bout = pickle.load(f)
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'cov1': tf.Variable(wc1),
        # 5x5 conv, 32 inputs, 64 outputs
        'cov2': tf.Variable(wc2),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'fc1': tf.Variable(wd1),
        # 1024 inputs, 10 outputs (class prediction)
        'fc2': tf.Variable(out)
    }

    biases = {
        'cov1': tf.Variable(bc1),
        'cov2': tf.Variable(bc2),
        'fc1': tf.Variable(bd1),
        'fc2': tf.Variable(bout)
    }
else:
    weights = {
        'cov1': tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1)),
        'cov2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),
        'fc1': tf.Variable(tf.random_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512])),
        'fc2': tf.Variable(tf.random_normal([512, NUM_LABELS]))
    }
    biases = {
        'cov1': tf.Variable(tf.random_normal([32])),
        'cov2': tf.Variable(tf.random_normal([64])),
        'fc1': tf.Variable(tf.random_normal([512])),
        'fc2': tf.Variable(tf.random_normal([10]))
    }
    weights_index = {
        'cov1': np.zeros([5, 5, NUM_CHANNELS, 32]),
        'cov2': np.zeros([5, 5, 32, 64]),
        'fc1': np.zeros([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512]),
        'fc2': np.zeros([512, NUM_LABELS])
    }
# Create model
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
# def quantize_a_value(val):
#
#
'''
mask gradients, for weights that are pruned, stop its backprop
'''
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
        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])
        keys = ['cov1','cov2','fc1','fc2']

        x_image = tf.reshape(x,[-1,28,28,1])
        # Construct model
        pred, pool = conv_network(x_image, weights, biases)

        # Define loss and optimizer
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        merged = tf.merge_all_summaries()
        saver = tf.train.Saver()

        # I need to fetch this value
        variables = [weights['cov1'], weights['cov2'], weights['fc1'], weights['fc2'],
                    biases['cov1'], biases['cov2'], biases['fc1'], biases['fc2']]
        org_grads = trainer.compute_gradients(cost, var_list = variables, gate_gradients = trainer.GATE_OP)
        org_grads = [(ClipIfNotNone(grad), var) for grad, var in org_grads]
        train_step = trainer.apply_gradients(org_grads)

        init = tf.initialize_all_variables()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # restore model if exists
            # if (os.path.isfile("tmp_20160118/model.meta")):
            #     op = tf.train.import_meta_graph("tmp_20160118/model.meta")
            #     op.restore(sess,tf.train.latest_checkpoint('tmp_20160118/'))
            #     print ("model found and restored")

            # print(weights['fc1'].eval())
            keys = ['cov1','cov2','fc1','fc2']
            weights_val = {}
            centroids = {}
            cluster_index = {}
            weights_orgs = {}
            biases_orgs = {}

            for key in keys:
                weight_org = weights[key].eval()
                weight= weight_org.flatten()
                weight_val = weight[weight != 0]
                # poses = [np.argwhere(weight_org == x) for x in weight_val]
                data = np.expand_dims(weight_val, axis = 1)
                # use kmeans to cluster
                kmeans = KMeans(n_clusters= Number_of_cluster, random_state=0).fit(data)
                centroid =  kmeans.cluster_centers_
                # add centroid value
                centroids[key] = centroid
                # add index value
                # indexs are stored in weight_org
                index = kmeans.labels_ + 1
                for w in np.nditer(weight_org, op_flags=['readwrite']):
                    if (w != 0):
                        w[...] = kmeans.predict(w)+1
                # sys.exit()
                cluster_index[key] = weight_org
                weights_orgs[key] = weights[key].eval()
                biases_orgs[key] = biases[key].eval()
            prune_info(weights,0)
            with open('quantize_info.pkl','wb') as f:
                pickle.dump((weights_orgs, biases_orgs, cluster_index,centroids),f)





        # KMeans(init='k-means++', n_clusters=16, n_init=10).fit(weights_val['cov1'])

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
