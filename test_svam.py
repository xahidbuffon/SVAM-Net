"""
# > Script for evaluating ** 
#    - Paper: 
"""
import os
import cv2
import time
import ntpath
import argparse
import numpy as np
from PIL import Image
from glob import glob
from os.path import join, exists
# keras libs
from keras.layers import Input
from keras.models import Model
# local libs
from models.svam_model import SVAM_Net
from utils.data_utils import preprocess, deprocess_mask

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


# input and output data shape
im_res, chans = (256, 256), 3
im_shape = (256, 256, 3)
x_in = Input(batch_shape=(1, im_res[1], im_res[0], chans))

def test(test_dir, res_dir, model_h5):
    ## create dir for output test data
    if not exists(res_dir): os.makedirs(res_dir)
    test_paths = sorted(glob(join(test_dir, "*.*")))
    print ("{0} test images are loaded".format(len(test_paths)))

    ## load specific model 
    assert os.path.exists(model_h5), "h5 model not found"
    model = SVAM_Net(res=im_shape)
    model.load_weights(model_h5)

    # testing loop
    times = []; s = time.time()
    for img_path in test_paths:
        # prepare data
        img_name = ntpath.basename(img_path).split('.')[0]
        inp_img = np.array(Image.open(img_path).resize(im_res))
        im = np.expand_dims(preprocess(inp_img), axis=0)        
        # generate saliency
        t0 = time.time()
        saml, sambu, samd, out = model.predict(im)
        times.append(time.time()-t0)
        _, out_bu, _, out_tdr = deprocess_gens(saml, sambu, samd, out, im_res)
        print ("tested: {0}".format(img_path))
        Image.fromarray(inp_img).save(join(res_dir, img_name+".jpg"))
        Image.fromarray(out_bu).save(join(res_dir, img_name+"_bu.png"))
        Image.fromarray(out_tdr).save(join(res_dir, img_name+"_tdr.png"))

    # some statistics    
    num_test = len(test_paths)
    if (num_test==0): print ("\nFound no images for test")
    else:
        print ("\nTotal images: {0}".format(num_test)) 
        # accumulate frame processing times (without bootstrap)
        Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
        print ("Time taken: {0} sec at {1} fps".format(Ttime, 1./Mtime))
        print("\nSaved generated images in in {0}\n".format(res_dir))


parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", type=str, default="data/test/")
parser.add_argument("--res_dir", type=str, default="data/output_svam/")
parser.add_argument("--ckpt_h5", type=str, default="checkpoints/SVAM_Net.h5")
args = parser.parse_args()
test(args.test_dir, args.res_dir, args.ckpt_h5)


