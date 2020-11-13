"""
# > Utility modules for handling data
"""
from __future__ import division
from __future__ import absolute_import
import os
import random
import fnmatch
import numpy as np
from PIL import Image
from os.path import join, isdir


def preprocess(x, band=(-1,1)):
    """
       - Transform input tensor x
       - image format [0,255] -> [band]  
    """
    if band==(-1, 1): # [0,255]->[-1,1]
        return (x / 127.5) - 1.0
    elif band==(0, 1): #[0,255]->[0,1]  
        return x / 255.0
    else: # mean-centered
        x = x.astype(np.float32)
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
        return x


def deprocess(x, band=(-1,1), unit8=True):
    """
       - Transform input tensor x
       - [band] -> image format [0, 255]  
    """
    if band==(-1, 1): # [-1,1]->[0,255]
        x = (x + 1.0) * 127.5
    elif band==(0, 1): # [0,1]->[0,255]  
        x = x * 255.0
    else: # mean-centered
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
    return np.uint8(x) if unit8 else x 


def preprocess_mask(x, normalize=False):
    """ Pre-process input tensor (of masks)
    """
    x = x/255.0
    x[x<0.2] = 0
    return np.expand_dims(x, axis=3)


def deprocess_mask(im, normalize=True, unit8=True):
    """ Post-process output mask
    """
    im[im<0.2] = 0
    if not normalize:
        im = im * 255.0
    else:
        max_im = (np.max(im) + 1e-8)
        im = im/max_im*255.0 # [0,1] -> [0,255]
    return np.uint8(im) if unit8 else im 


def read_and_resize(path, res=(256, 256), mode_='RGB'):
    """
       - Get image from given path and reshape to res
       - Return as array  
    """
    im = Image.open(path).resize(res)
    if im.mode=='L' and mode_=='RGB': 
        copy = np.zeros((res[1], res[0], 3))
        copy[:, :, 0] = im
        copy[:, :, 1] = im
        copy[:, :, 2] = im
        im = copy
    elif mode_=='L':  
        im = im.convert("L")
    return np.array(im).astype(np.float32)


class dataLoaderSOD():
    """
       - Data loader for UFO dataset <image, mask, edge>
       - Does training-validation splits
       - Pipelines the data into tensors
    """
    def __init__(self, data_path, dataset, res):
        self.im_res = res
        if isdir(data_path): 
            self.hr_folder, self.mask_folder = self.get_sub_folders(dataset)
            self.get_train_and_val_paths(data_path)
        elif data_path.endswith(".txt"):
            self.get_train_and_val_pairs_txt_file(data_path)
        else:pass

    def get_sub_folders(self, dataset):
        if dataset=="UFO-120": return "train_val/hr/", "train_val/mask/"
        elif dataset=="DUTS":  return "DUTS_TR/Images/", "DUTS_TR/Masks/"
        else: return "images/", "masks/"     

    def get_train_and_val_pairs_txt_file(self, txt_file):
        self.num_train, self.num_val = 0, 0
        self.train_hr_paths, self.val_hr_paths  = [], []
        self.train_mask_paths, self.val_mask_paths =  [], []
        with open(txt_file) as f:
            lines = [line.rstrip() for line in f]
        hr_path, mask_path = [], [] 
        for line in lines:
            im_p, m_p = line.split(',')
            hr_path.append(im_p) 
            mask_path.append(m_p)
        # generate paired data paths 
        num_paths = min(len(hr_path), len(mask_path))
        all_idx = range(num_paths)
        # 95% train-val splits
        random.shuffle(all_idx)
        self.num_train = int(num_paths*0.95)
        self.num_val = num_paths-self.num_train
        train_idx = set(all_idx[:self.num_train])
        # split data paths to training and validation sets
        for i in range(num_paths):
            if i in train_idx:
                self.train_hr_paths.append(hr_path[i]) 
                self.train_mask_paths.append(mask_path[i])
            else:
                self.val_hr_paths.append(hr_path[i]) 
                self.val_mask_paths.append(mask_path[i])
        print ("Loaded {0} samples for training".format(self.num_train))
        print ("Loaded {0} samples for validation".format(self.num_val)) 

    def get_train_and_val_paths(self, data_dir):
        self.num_train, self.num_val = 0, 0
        self.train_hr_paths, self.val_hr_paths  = [], []
        self.train_mask_paths, self.val_mask_paths =  [], []
        # generate paired data paths 
        hr_path = sorted(os.listdir(data_dir+self.hr_folder))
        mask_path = sorted(os.listdir(data_dir+self.mask_folder))
        num_paths = min(len(hr_path), len(mask_path))
        all_idx = range(num_paths)
        # 95% train-val splits
        random.shuffle(all_idx)
        self.num_train = int(num_paths*0.95)
        self.num_val = num_paths-self.num_train
        train_idx = set(all_idx[:self.num_train])
        # split data paths to training and validation sets
        for i in range(num_paths):
            if i in train_idx:
                self.train_hr_paths.append(data_dir+self.hr_folder+hr_path[i]) 
                self.train_mask_paths.append(data_dir+self.mask_folder+mask_path[i])
            else:
                self.val_hr_paths.append(data_dir+self.hr_folder+hr_path[i]) 
                self.val_mask_paths.append(data_dir+self.mask_folder+mask_path[i])
        print ("Loaded {0} samples for training".format(self.num_train))
        print ("Loaded {0} samples for validation".format(self.num_val)) 

    def load_batch(self, batch_size=1):
        self.n_batches = self.num_train//batch_size
        for i in range(self.n_batches-1):
            batch_hr = self.train_hr_paths[i*batch_size:(i+1)*batch_size]
            batch_m = self.train_mask_paths[i*batch_size:(i+1)*batch_size]
            # make pairs and yield
            imgs_lr, imgs_mask = [], []
            for idx in range(len(batch_hr)):
                img_lr = read_and_resize(batch_hr[idx], res=self.im_res)
                img_mask = read_and_resize(batch_m[idx], res=self.im_res, mode_='L')
                imgs_lr.append(img_lr)
                imgs_mask.append(img_mask)
            imgs_lr = preprocess(np.array(imgs_lr))
            imgs_mask = preprocess_mask(np.array(imgs_mask))
            yield imgs_lr, imgs_mask

    def load_val_data(self, batch_size=1):
        idx = np.random.choice(np.arange(self.num_val), batch_size, replace=False)
        paths_hr = [self.val_hr_paths[i] for i in idx]
        paths_mask = [self.val_mask_paths[i] for i in idx]
        imgs_lr, imgs_mask = [], []
        for idx in range(len(paths_hr)):
            img_lr = read_and_resize(paths_hr[idx], res=self.im_res)
            img_mask = read_and_resize(paths_mask[idx], res=self.im_res, mode_='L')
            imgs_lr.append(img_lr)
            imgs_mask.append(img_mask)
        imgs_lr = preprocess(np.array(imgs_lr))
        imgs_mask = preprocess_mask(np.array(imgs_mask))
        return imgs_lr, imgs_mask


def getTrainGenerator(loader, batch_size=1, output_head=1):
    n_batches = loader.num_train//batch_size
    while True:
        for i in range(n_batches-1):
            imgs_path = loader.train_hr_paths[i*batch_size:(i+1)*batch_size]
            masks_path = loader.train_mask_paths[i*batch_size:(i+1)*batch_size]
            # make pairs and yield
            imgs_, masks_ = [], []
            for idx in range(len(imgs_path)):
                img = read_and_resize(imgs_path[idx], res=loader.im_res)
                mask = read_and_resize(masks_path[idx], res=loader.im_res, mode_='L')
                imgs_.append(img)
                masks_.append(mask)
            imgs_ = preprocess(np.array(imgs_))
            masks_ = preprocess_mask(np.array(masks_))
            if output_head==1: yield imgs_, masks_
            else: yield imgs_, [masks_]*output_head

