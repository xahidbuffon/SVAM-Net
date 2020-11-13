"""
# > The end-to-end SVAM-Net architecture 
"""
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.models import model_from_json
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import Activation, Add, Multiply, Concatenate, BatchNormalization
from bilinear_upsampling import BilinearUpsampling
from vgg_en_dec import plain_vgg_en_dec


def BN(input_tensor, block_name):
    bn = BatchNormalization(name = block_name + '_BN')(input_tensor)
    ac = Activation('relu', name = block_name + '_relu')(bn)
    return ac


def SAM_TopDown(dec1):
    # block 0
    b1_pred = Conv2D(128, (3, 3), strides=1, padding='same', name='b1_pred')(dec1)
    samtd_logit = Conv2D(1, (3, 3), strides=1, padding='same', name='samd_logit')(b1_pred)
    b1_conv = BN(b1_pred, 'b1_conv')
    # RRM block 1
    b2_pred = Conv2D(128, (3, 3), strides=1, padding='same', name='b2_pred')(b1_conv)
    b2_conv = BN(b2_pred, 'b2_conv')
    b2_cont = Concatenate(name='b2c')([b2_conv, b1_conv]) 
    # RRM block 2
    b3_pred = Conv2D(128, (3, 3), strides=1, padding='same', name='b3_pred')(b2_cont)
    b3_conv = BN(b3_pred, 'b3_conv')
    b3_cont = Concatenate(name='b3_cont')([b3_conv, b2_conv]) 
    # RR addition
    b4 = Conv2D(128, (3, 3), strides=1, padding='same', name='b4')(b3_cont)
    samd_res = Add(name='samd_res')([b4, b1_conv])
    samtdr_out = Conv2D(1, (3, 3), padding='same', activation='sigmoid', name='samd_out')(samd_res)
    return samtd_logit, samtdr_out


def SAM_Aux(conv22, conv33):
    saml_l33 = Conv2D(512, (3, 3), padding='same', activation='relu', name='saml_l33')(conv33)
    saml_l22 = Conv2D(256, (3, 3), padding='same', activation='relu', name='saml_l22')(conv22)
    saml_conva = BilinearUpsampling(upsampling=(4, 4), name='saml_conva')(saml_l33)
    saml_convb = BilinearUpsampling(upsampling=(2, 2), name='saml_convb')(saml_l22)
    saml_convc = Concatenate(name='saml_convc')([saml_conva, saml_convb])
    saml_conv = Conv2D(128, (3, 3), padding='same', name='saml_conv')(saml_convc)
    saml_aux = Conv2D(1, (3, 3), padding='same', activation='sigmoid', name='saml_out')(saml_conv)
    return saml_aux


def SAM_BottomUp(pool4, conv53):
    sam_bo1 = Add(name='sam_bo1')([pool4, conv53])
    sam_bo2 = BilinearUpsampling(upsampling=(4, 4), name='sam_bo2')(sam_bo1)
    sam_bo3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='sam_bo3')(sam_bo2)
    sam_bo4 = Conv2D(256, (3, 3), padding='same', activation='relu', name='sam_bo4')(sam_bo3)
    sam_bo5 = BilinearUpsampling(upsampling=(4, 4), name='sam_bo5')(sam_bo4)
    sambu_logit = Conv2D(1, (3, 3), strides=1, padding='same', name='sambo_logit')(sam_bo5)
    return sambu_logit


def SVAM_Net(res=(256, 256, 3), model_h5=None):
    model = plain_vgg_en_dec(res)
    if model_h5: model.load_weights(model_h5) 
    model.trainable = True
    for layer in model.layers:
        layer.trainable = True
    # pre-trained encoder layers
    conv12, pool1 = model.get_layer('block1_conv2').output, model.get_layer('block1_pool').output
    conv22, pool2 = model.get_layer('block2_conv2').output, model.get_layer('block2_pool').output
    conv33, pool3 = model.get_layer('block3_conv3').output, model.get_layer('block3_pool').output
    conv43, pool4 = model.get_layer('block4_conv3').output, model.get_layer('block4_pool').output
    conv53, pool5 = model.get_layer('block5_conv3').output, model.get_layer('block5_pool').output
    # pre-trained deecoder layers
    conv2d_1, dec5 = model.get_layer('conv2d_1').output, model.get_layer('concatenate_1').output
    conv2d_2, dec4 = model.get_layer('conv2d_2').output, model.get_layer('concatenate_2').output
    conv2d_3, dec3 = model.get_layer('conv2d_3').output, model.get_layer('concatenate_3').output
    conv2d_4, dec2 = model.get_layer('conv2d_4').output, model.get_layer('concatenate_4').output
    dec1, dec0 = model.get_layer('up_sampling2d_5').output, model.get_layer('output').output
    # get SAM outputs
    saml_aux = SAM_Aux(conv22, conv33)
    sambu_logit = SAM_BottomUp(pool4, conv53)
    samtd_logit, samtdr_out = SAM_TopDown(dec1)
    return Model(model.input, [saml_aux, sambu_logit, samtd_logit, samtdr_out])
    
