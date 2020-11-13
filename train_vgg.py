"""
# > Pre-training script for the VGG backbone encoder-decoder 
#    - Paper: ***
"""
from __future__ import print_function, division
import os
import yaml
import math
import argparse
from PIL import Image
from os.path import join, exists
# keras libs
from keras import callbacks
from keras.optimizers import Adam, SGD
# local libs
from utils.loss_utils import accuracy
from models.vgg_en_dec import plain_vgg_en_dec
from utils.data_utils import dataLoaderSOD, getTrainGenerator
#####################################################################


def train(cfg):
    # dataset info
    dataset = cfg["dataset_name"] 
    data_path = cfg["dataset_path"]
    # image info
    chans = cfg["channels"]
    im_res = (cfg["im_width"], cfg["im_height"]) 
    # training params
    num_epochs = cfg["num_epochs"]
    batch_size =  cfg["batch_size"]
    # create validation and checkpoint directories
    ckpt_dir = join("checkpoints/", dataset+'_vgg')
    if not exists(ckpt_dir): os.makedirs(ckpt_dir)

    base_lr = 1e-2
    optimizer_ = SGD(lr=base_lr, momentum=0.9, decay=0)
    loss_ = ['binary_crossentropy']
    metrics_ = [accuracy]

    # create model and compile
    model = plain_vgg_en_dec(res=(im_res[1], im_res[0], chans))
    model_fname = join(ckpt_dir, 'model_pt.h5')
    load_pretrained = False 
    if load_pretrained: 
        assert exists(model_fname)
        model.load_weights(model_fname)

    data_loader = dataLoaderSOD(data_path, dataset, im_res)
    steps_per_epoch_ = (data_loader.num_train//batch_size)
    train_generator = getTrainGenerator(data_loader, batch_size)

    def lr_scheduler(epoch):
        drop = 0.5
        epoch_drop = num_epochs / 8.0
        lr = base_lr * math.pow(drop, math.floor(( 1 + epoch) / epoch_drop))
        return lr

    lr_decay = callbacks.LearningRateScheduler(schedule = lr_scheduler)
    modelcheck = callbacks.ModelCheckpoint(
                                           model_fname, 
                                           monitor = 'loss', 
                                           save_best_only = True, 
                                           save_weights_only = True, 
                                           period = 1,
                                           verbose = 1, 
                                           mode= 'auto'
                                          )

    model.compile(optimizer=optimizer_, loss=['binary_crossentropy'], metrics =['accuracy'])
    model.fit_generator(
                        train_generator, 
                        steps_per_epoch = steps_per_epoch_,
                        epochs = num_epochs, 
                        verbose = 1,
                        callbacks = [lr_decay, modelcheck]
                       )


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="configs/train_all.yaml")
args = parser.parse_args()
# load the configuration file
with open(args.cfg_file) as f:
    cfg = yaml.load(f)
train(cfg)

