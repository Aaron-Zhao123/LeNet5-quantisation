import os
import cdfp_training
import pickle
import numpy as np

acc_list = []
count = 0
pcov = 0
pfc = 0
pcov2 = 0
pfc2 = 0
base_name = 'base'
learning_rate = 1e-5
quantisation_bits = [2,4,8,16,32,64]
# quantisation_bits = [32]
pre_train_acc_list = []
test_acc_list = []
with open('weights/base.pkl', 'rb') as f:
    wc1, wc2, wd1, out, bc1, bc2, bd1, bout = pickle.load(f)
weights_val = {
    'cov1': wc1,
    'cov2': wc2,
    'fc1': wd1,
    'fc2': out
}

keys = ['cov1', 'cov2', 'fc1', 'fc2']
central_value = {}
c_pos = {}
c_neg = {}

for key in keys:
    central_value[key] = np.mean(weights_val[key])
    c_pos[key] = np.mean(weights_val[key][weights_val[key] > central_value[key]])
    c_neg[key] = np.mean(weights_val[key][weights_val[key] <= central_value[key]])



for q_width in quantisation_bits:
    # set Parameters
    param = [
    ('-quantisation_bits', q_width),
    ('-parent_dir', './'),
    ('-base_name', base_name),
    ('-d_range', 4),
    ('-c_pos', c_pos),
    ('-c_neg', c_neg),
    ]
    (pre_acc, test_acc) = dfp_training.main(param)
    print('pre train acc is {}, after train acc is {}'.format(pre_acc, test_acc))
    pre_train_acc_list.append(pre_acc)
    test_acc_list.append(test_acc)

print(pre_train_acc_list)
print(test_acc_list)
with open('acc.txt','wb') as f:
    f.write(pre_train_acc_list)
    f.write(test_acc_list)
