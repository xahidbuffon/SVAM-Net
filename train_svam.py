"""
# > Training script for SVAM-Net 
#    - Paper: https://arxiv.org/pdf/2011.06252.pdf 
"""
import os
import yaml
import argparse
import numpy as np
from PIL import Image
from os.path import join, exists
# keras libs
import tensorflow as tf
from keras import callbacks
from keras.optimizers import Adam
# local libs
from models.svam_model import SVAM_Net
from utils.loss_utils import EdgeHoldLoss
from utils.data_utils import dataLoaderSOD
from utils.data_utils import deprocess, deprocess_mask

def sigmoid(x):
    """ Numerically stable sigmoid
    """
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x))
                   )

def deprocess_gens(saml, sambu, samtd, out, im_res):
    """ Post-process all outputs
    """
    samtd, sambu =  sigmoid(samtd), sigmoid(sambu)
    out = deprocess_mask(out).reshape(im_res) 
    saml = deprocess_mask(saml).reshape(im_res) 
    samtd = deprocess_mask(samtd).reshape(im_res) 
    sambu = deprocess_mask(sambu).reshape(im_res)
    return saml, sambu, samtd, out


def train(cfg):
    """ Training pipeline
         - cfg: yaml file with trainig parameters (see configs/)
    """
    # dataset info
    dataset = cfg["dataset_name"] 
    data_path = cfg["dataset_path"]
    # image info
    chans = cfg["channels"]
    im_res = (cfg["im_width"], cfg["im_height"])  
    im_shape = (im_res[1], im_res[0], chans) 
    # training params
    num_epochs = cfg["num_epochs"]
    batch_size =  cfg["batch_size"]
    val_interval = cfg["val_interval"]
    ckpt_interval = cfg["ckpt_interval"]
    # create validation and checkpoint directories
    val_dir = join("samples/", dataset+'_usal')
    if not exists(val_dir): os.makedirs(val_dir)
    ckpt_dir = join("checkpoints/", dataset+'_usal')
    if not exists(ckpt_dir): os.makedirs(ckpt_dir)

    ## data pipeline
    data_loader = dataLoaderSOD(data_path, dataset, im_res)
    steps_per_epoch = (data_loader.num_train//batch_size)
    num_step = num_epochs * steps_per_epoch

    ## define model, load pretrained weights
    model = SVAM_Net(model_h5='checkpoints/vgg16_ed_pt.h5')

    ## compile model
    model.compile(
                  optimizer = Adam(3e-4, 0.5), 
                  loss = ['binary_crossentropy', EdgeHoldLoss, EdgeHoldLoss, 'binary_crossentropy'], 
                  loss_weights = [0.5, 1, 2, 4],
                  metrics =['accuracy']
                 )

    ## setup training pipeline and fit model
    print ("\nTraining SVAM-Net...")
    it, step, epoch = 1, 1, 1
    while (step <= num_step):
        for imgs, masks in data_loader.load_batch(batch_size):
            loss = model.train_on_batch(imgs, [masks, masks, masks, masks])
            # increment step, and show the progress 
            it += 1; step += 1;
            if not step%100: 
               print ("Epoch {0}:{1}/{2}. Loss: {3}".format(epoch, step, num_step, loss[0]))
            ## validate and save samples at regular intervals
            if (step % val_interval==0):
                inp_img, gt_sal = data_loader.load_val_data(batch_size=1)
                saml, sambu, samtd, out = model.predict(inp_img)
                inp_img = deprocess(inp_img).reshape(im_shape)
                saml, sambu, samtd, out = deprocess_gens(saml, sambu, samtd, out, im_res)
                Image.fromarray(inp_img).save(join(val_dir, "%d_in.png" %step))
                Image.fromarray(np.hstack((saml,sambu,samtd,out))).save(join(val_dir, "%d_sal.png" %step))
        epoch += 1; it = 0;
        ## save model at regular intervals
        if epoch % ckpt_interval==0:
            model.save_weights(join(ckpt_dir, ("model_%d.h5" %epoch)))
            print("\nSaved model in {0}\n".format(ckpt_dir))



##################################
parser = argparse.ArgumentParser()
#parser.add_argument("--cfg_file", type=str, default="../configs/train_all.yaml")
parser.add_argument("--cfg_file", type=str, default="../configs/train_ufo.yaml")
args = parser.parse_args()
# load the configuration file
with open(args.cfg_file) as f:
    cfg = yaml.load(f)
train(cfg)

