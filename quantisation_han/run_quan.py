import quantisation_training
import Quantisation_v3

def main():
    acc_list = []
    pre_acc_list = []
    clusters = [2,4,8,16,32,64]
    count = 0
    retrain = 0
    for cluster in clusters:
        param = [
            ('-cluster',cluster)
            ]
        Quantisation_v3.main(param)
        pre_acc, acc = quantisation_training.main(param)
        pre_acc_list.append(pre_acc)
        acc_list.append(acc)
        print (acc)
        count = count + 1
        # if (acc >= 0.9936 or retrain >=3):
        #     acc_list.append(acc)
        #     retrain = 0
        #     count = count + 1
        # else:
        #     retrain += 1
    print('accuracy summary: {}'.format(acc_list))
    print(pre_acc_list)
    print(acc_list)



if __name__ == '__main__':
    main()
