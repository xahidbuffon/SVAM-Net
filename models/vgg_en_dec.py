"""
# > VGG backbone encoder-decoder model 
     - used for the pre-training step in SVAM-Net
"""
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Dropout, Concatenate
from keras.layers import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications.vgg16 import VGG16


def myUpSample2X(layer_input, skip_input, filters, f_size=3, dropout_rate=0):
    ## for upsampling
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    if dropout_rate: u = Dropout(dropout_rate)(u)
    u = BatchNormalization(momentum=0.8)(u)
    u = Concatenate()([u, skip_input])
    return u


def plain_vgg_en_dec(res=(256, 256, 3)):
    vgg = VGG16(input_shape=res, include_top=False, weights='imagenet')
    vgg.trainable = True
    for layer in vgg.layers:
        #print (layer.name)
        layer.trainable = True
    # encoder network
    conv12, pool1 = vgg.get_layer('block1_conv2').output, vgg.get_layer('block1_pool').output
    conv22, pool2 = vgg.get_layer('block2_conv2').output, vgg.get_layer('block2_pool').output
    conv33, pool3 = vgg.get_layer('block3_conv3').output, vgg.get_layer('block3_pool').output
    conv43, pool4 = vgg.get_layer('block4_conv3').output, vgg.get_layer('block4_pool').output
    conv53, pool5 = vgg.get_layer('block5_conv3').output, vgg.get_layer('block5_pool').output
    # encoder network
    dec5 = myUpSample2X(pool5,pool4, 512)
    dec4 = myUpSample2X(dec5, pool3, 512)
    dec3 = myUpSample2X(dec4, pool2, 256)
    dec2 = myUpSample2X(dec3, pool1, 128)
    dec1 = UpSampling2D(size=2)(dec2)
    out = Conv2D(1, (1, 1), strides=1, padding='same', activation='sigmoid', name='output')(dec1)
    return Model(vgg.input, out)
    
