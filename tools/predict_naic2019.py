
import os
import json
import argparse
import pickle
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./save_for_test.pkl')
    parser.add_argument('--save', default='./submission.json')
    parser.add_argument('--rank', type=int, default=200)

    args = parser.parse_args()
    return args


def main(args):

    test_info = pickle.load(open(args.data, 'rb'))
    query = test_info['query']
    gallery = test_info['gallery']
    distmat = test_info['distmat']

    submission = {}
    for i, (img_path, pid, camid) in enumerate(query):
        imgname = img_path.split('/')[-1]
        dist = distmat[i]
        ids = np.argsort(dist)
        submission[imgname] = []
        for j in range(args.rank):
            matchedname = gallery[ids[j]][0].split('/')[-1]
            submission[imgname].append(matchedname)

    with open(args.save, 'w') as w_obj:
        json.dump(submission, w_obj)

    print("Results have been saved to {}".format(os.path.abspath(args.save)))
    

if __name__ == "__main__":
    args = parse_args()
    main(args)