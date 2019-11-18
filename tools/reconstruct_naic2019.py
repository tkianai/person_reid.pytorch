
import os
import argparse
import random
import shutil
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./datasets/naic2019')
    parser.add_argument('--ratio', type=float, default=0.2)
    parser.add_argument('--save', default=None)
    args = parser.parse_args()

    if args.save is None:
        args.save = args.root
    return args


def main(args):

    os.system("unzip {} -d {}".format(os.path.join(args.root, '初赛训练集.zip'), args.root))
    os.system("unzip {} -d {}".format(os.path.join(args.root, '初赛A榜测试集.zip'), args.root))

    train_labels = {}
    with open(os.path.join(args.root, '初赛训练集', 'train_list.txt')) as r_obj:
        for line in r_obj:
            if line.strip():
                img_path, pid = line.split(' ')
                if int(pid) not in train_labels:
                    train_labels[int(pid)] = []
                train_labels[int(pid)].append(img_path)
    
    pids = list(train_labels.keys())
    random.shuffle(pids)
    val_num = int(len(pids) * args.ratio)
    val_set = set(pids[:val_num])
    print("There are {} identities in total.".format(len(pids)))
    print("Using raw {} identities as evaluation set.".format(val_num))

    # save train set
    train_dir = os.path.join(args.save, 'train')
    if os.path.exists(train_dir):
        raise "There exists train dir, please remove it first"

    os.system("mv {} {}".format(os.path.join(
        args.root, '初赛训练集', 'train_list.txt'), os.path.join(args.save, 'train_list_total.txt')))
    os.system("mv {} {}".format(os.path.join(args.root, "初赛训练集", 'train_set'), train_dir))

    train_list = open(os.path.join(args.save, 'train_list.txt'), 'w')
    query_list = open(os.path.join(args.save, 'query_list.txt'), 'w')
    gallery_list = open(os.path.join(args.save, 'gallery_list.txt'), 'w')

    valid_val_num = 0
    for key, value in train_labels.items():
        # NOTE filter the single one image identity
        if key in val_set and len(value) > 1:
            valid_val_num += 1
            query_list.write(value[0] + ' ' + str(key) + '\n')
            for i in range(1, len(value)):
                gallery_list.write(value[i] + ' ' + str(key) + '\n')
        else:
            for i in range(len(value)):
                train_list.write(value[i] + ' ' + str(key) + '\n')

    print("Using {} identities as a evaluation set.".format(valid_val_num))
    train_list.close()
    query_list.close()
    gallery_list.close()

    # handle test set
    test_dir = os.path.join(args.save, 'test')
    if os.path.exists(test_dir):
        raise "There exists test dir, please remove it first"
    else:
        os.makedirs(test_dir)
    
    # handle test_a
    test_a_dir = os.path.join(test_dir, 'test_a')
    os.system("mv {} {}".format(os.path.join(args.root, "初赛A榜测试集"), test_a_dir))
    gallery_a_list = open(os.path.join(test_a_dir, 'gallery_a_list.txt'), 'w')
    for filename in os.listdir(os.path.join(test_a_dir, 'gallery_a')):
        gallery_a_list.write('gallery_a/' + filename + '\n')
    gallery_a_list.close()

    # delete the unused data
    os.system("rm -rf {}".format(os.path.join(args.root, '初赛训练集')))
    os.system("rm -rf {}".format(os.path.join(args.root, '__MACOSX')))
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
