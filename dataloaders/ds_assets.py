import torchvision.transforms as transforms
import torch
import yaml
import time
import os
import os.path as osp
import os.path as op
import shutil
import torch.backends.cudnn as cudnn
import pickle
import argparse
from tqdm import tqdm
import sys
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
import re
import math
from dataloaders.loader_base import *
from typing import List
from utils.iotools import *


def _process_dir_market(dir_path='/export/livia/home/vision/Rbhattacharya/work/data/data/market1501/Market-1501-v15.09.15/bounding_box_test', relabel=False):
    img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')

    pid_container = set()
    for img_path in sorted(img_paths):
        pid, _ = map(int, pattern.search(img_path).groups())
        if pid == -1: continue  # junk images are just ignored
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    dataset = []
    for img_path in sorted(img_paths):
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1: continue  # junk images are just ignored
        if pid == 0: continue  # junk images are just ignored
        assert 0 <= pid <= 1501  # pid == 0 means background
        assert 1 <= camid <= 6
        camid -= 1  # index starts from 0
        if relabel: pid = pid2label[pid]

        dataset.append((img_path, pid, camid, 0))
    return dataset

def _process_dir_duke(dir_path, relabel=False, pid_begin=0):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        cam_container = set()
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid_begin + pid, camid, 0))
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        return dataset

def _process_dir_msmt(dir_path, list_path, exclude_after=-1, pid_begin=0):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            if exclude_after != -1: # added exclusion pids
                if pid > exclude_after: continue
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid_begin+pid, camid-1, 0))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset


class RSTPReid(BaseDataset):
    """
    RSTPReid

    Reference:
    DSSL: Deep Surroundings-person Separation Learning for Text-based Person Retrieval MM 21

    URL: http://arxiv.org/abs/2109.05534

    Dataset statistics:
    # identities: 4101 
    """
    dataset_dir = 'RSTPReid'

    def __init__(self, root='', verbose=True):
        super(RSTPReid, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')

        self.anno_path = op.join(self.dataset_dir, 'data_captions.json')
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            print("=> RSTPReid Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

  
    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['img_path'])
                captions = anno['captions'] # caption list
                for caption in captions:
                    dataset.append((pid, image_id, img_path, caption))
                image_id += 1
            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['img_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions'] # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))



class CUHKPEDES(BaseDataset):
    """
    CUHK-PEDES

    Reference:
    Person Search With Natural Language Description (CVPR 2017)

    URL: https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Person_Search_With_CVPR_2017_paper.html

    Dataset statistics:
    ### identities: 13003
    ### images: 40206,  (train)  (test)  (val)
    ### captions: 
    ### 9 images have more than 2 captions
    ### 4 identity have only one image

    annotation format: 
    [{'split', str,
      'captions', list,
      'file_path', str,
      'processed_tokens', list,
      'id', int}...]
    """
    dataset_dir = 'CUHK-PEDES'

    def __init__(self, root='', verbose=True):
        super(CUHKPEDES, self).__init__()
        root = "/export/livia/home/vision/Rbhattacharya/work/data/CUHK_PEDES/"
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')

        self.anno_path = op.join(self.dataset_dir, 'reid_raw.json')
        self.anno_path = '/export/livia/home/vision/Rbhattacharya/work/data/CUHK_PEDES/CUHK-PEDES/r_cuhk_1110_dec_1110.json'
  
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            print("=> CUHK-PEDES Images and Captions are loaded")
            self.show_dataset_info()

    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

  
    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id']) - 1 # make pid begin from 0
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                captions = anno['captions'] # caption list
                dec_captions =  anno['diversity_captions'] 
                # captions = anno['mllm_captions'] # caption list
                for i, caption in enumerate(captions):
                    dataset.append((pid, image_id, img_path, caption, dec_captions[i], int(anno['sample_id'])))
                image_id += 1
            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            dec_captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno['id'])
                sample_id = int(anno['sample_id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions'] # caption list
                diversity_captions_list =  anno['val_diversity_captions'] 
                for i, caption in enumerate(caption_list):
                    captions.append(caption)
                    dec_captions.append(diversity_captions_list[i])
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions,
                "dec_captions": dec_captions,
                "sample_id": sample_id
            }

            return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
