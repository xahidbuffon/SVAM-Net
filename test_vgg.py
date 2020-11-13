"""
# > Script for evaluating ** 
#    - Paper: 
"""
import os
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
from models.vgg_en_dec import plain_vgg_en_dec
from utils.data_utils import preprocess, deprocess, deprocess_mask

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
    model = plain_vgg_en_dec(res=(im_res[1], im_res[0], chans))
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
        sal = model.predict(im)
        times.append(time.time()-t0)
        sal = deprocess_mask(sal, unit8=True).reshape(im_res) 
        Image.fromarray(inp_img).save(join(res_dir, img_name+".jpg"))
        Image.fromarray(sal).save(join(res_dir, img_name+"_sal.png"))
        print ("tested: {0}".format(img_path))

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
parser.add_argument("--res_dir", type=str, default="data/output_vgg16_ed/")
parser.add_argument("--ckpt_h5", type=str, default="checkpoints/vgg16_ed_pt.h5")
args = parser.parse_args()
test(args.test_dir, args.res_dir, args.ckpt_h5)


