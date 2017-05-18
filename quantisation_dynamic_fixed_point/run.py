import os
import dfp_training
acc_list = []
count = 0
pcov = 0
pfc = 0
pcov2 = 0
pfc2 = 0
base_name = 'base'
base_name = 'base_pruned'
learning_rate = 1e-5
# quantisation_bits = [2,4,8,16,32,64]
# quantisation_bits = [8,16,32,64]
dynamic_width = [2,4,8,16,32,64]
dynamic_width = [1,2,4,8,16]
quantisation_bits = [4,8,16,32]
quantisation_bits = [6]
quantisation_bits = [item-3 for item in quantisation_bits]
pre_train_acc_list = []
test_acc_list = []
for q_width in quantisation_bits:
# for d_width in dynamic_width:
    # set Parameters
    param = [
    ('-quantisation_bits', q_width),
    ('-parent_dir', './'),
    ('-base_name', base_name),
    ('-d_range', 4)
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
