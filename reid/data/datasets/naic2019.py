from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os.path as osp
import glob
import re
import warnings

from reid.data.datasets import ImageDataset


class NAIC2019(ImageDataset):
    """NAIC1501.
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """

    dataset_dir = 'naic2019'
    dataset_url = None

    def __init__(self, root='./datasets', naic2019_phase='train', naic2019_total=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        _train_list_file = 'train_list_total.txt' if naic2019_total else 'train_list.txt'
        train_list_path = osp.join(self.dataset_dir, _train_list_file)
        train = self.process_dir(self.dataset_dir, train_list_path, relabel=True)
        if naic2019_phase == 'train':
            query_list_path = osp.join(self.dataset_dir, 'query_list.txt')
            gallery_list_path = osp.join(self.dataset_dir, 'gallery_list.txt')
            query = self.process_dir(self.dataset_dir, query_list_path, relabel=False, flag='query')
            gallery = self.process_dir(self.dataset_dir, gallery_list_path, relabel=False, flag='gallery')
        else:
            self.test_dir = osp.join(self.dataset_dir, 'test/test_a')
            query_list_path = osp.join(self.test_dir, 'query_a_list.txt')
            gallery_list_path = osp.join(self.test_dir, 'gallery_a_list.txt')
            query = self.process_dir(self.test_dir, query_list_path, relabel=False, flag='query')
            gallery = self.process_dir(self.test_dir, gallery_list_path, relabel=False, flag='gallery')

        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, list_path, relabel=False, flag='query'):

        with open(list_path, 'r') as r_obj:
            data_list = r_obj.readlines()
        
        if relabel:
            pid_container = set()
            for itm in data_list:
                img_path, pid = itm.strip().split(' ')
                pid_container.add(int(pid))
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for itm in data_list:
            itm = itm.strip().split(' ')
            if len(itm) == 1:
                pid = 0
            else:
                pid = int(itm[1])
            camid = 0 if flag == 'query' else 1
            img_path = osp.join(dir_path, itm[0].strip())

            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data
