import os
import cdfp_training
import pickle
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import pylab

acc_list = []
count = 0
pcov = 0
pfc = 0
pcov2 = 0
pfc2 = 0
base_name = 'base'
learning_rate = 1e-5
quantisation_bits = [4,6,8,16,32]
quantisation_bits = [item-3 for item in quantisation_bits]
# quantisation_bits = [32]
pre_train_acc_list = []
test_acc_list = []
PLOT = 1
with open('weights/base.pkl', 'rb') as f:
    wc1, wc2, wd1, out, bc1, bc2, bd1, bout = pickle.load(f)
with open('masks/base.pkl', 'rb') as f:
    weights_mask = pickle.load(f)
weights_val = {
    'cov1': wc1,
    'cov2': wc2,
    'fc1': wd1,
    'fc2': out
}


if (PLOT):
    df = (weights_val['fc1'] * weights_mask['fc1']).flatten()
    df = df[df!=0]
    n, bins, patches = plt.hist(df, 100, normed=1, facecolor='blue', alpha=0.75)
    plt.grid(True)
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.savefig("test.png")



keys = ['cov1', 'cov2', 'fc1', 'fc2']
central_value = {}
c_pos = {}
c_neg = {}

for key in keys:
    central_value[key] = np.mean(weights_val[key]* weights_mask[key])
    pos_plus = np.logical_and((weights_val[key]* weights_mask[key]) > central_value[key], weights_mask[key])
    pos_minus = np.logical_and((weights_val[key]* weights_mask[key]) <= central_value[key], weights_mask[key])
    c_pos[key] = np.mean(weights_val[key][pos_plus])
    c_neg[key] = np.mean(weights_val[key][pos_minus])

print(central_value)
print(c_pos)
print(c_neg)
# sys.exit()


for q_width in quantisation_bits:
    # set Parameters
    param = [
    ('-quantisation_bits', q_width),
    ('-parent_dir', './'),
    ('-base_name', base_name),
    ('-d_range', 4),
    ('-c_pos', c_pos),
    ('-c_neg', c_neg),
    ('-central_value', central_value)
    ]
    (pre_acc, test_acc) = cdfp_training.main(param)
    print('pre train acc is {}, after train acc is {}'.format(pre_acc, test_acc))
    pre_train_acc_list.append(pre_acc)
    test_acc_list.append(test_acc)

print(pre_train_acc_list)
print(test_acc_list)
with open('acc.txt','wb') as f:
    f.write(pre_train_acc_list)
    f.write(test_acc_list)
