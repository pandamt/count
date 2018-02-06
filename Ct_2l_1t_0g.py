#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import random
from scipy import misc
import time
import sys
import load_count
from model_settings import learning_rate, batch_size, glimpses, img_height, img_width, p_size, min_edge, max_edge, min_blobs_train, max_blobs_train, min_blobs_test, max_blobs_test # MT

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if not os.path.exists("model_runs"):
    os.makedirs("model_runs")

if sys.argv[1] is not None:
        model_name = sys.argv[1]

folder_name = "model_runs/" + model_name

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

start_restore_index = 0

sys.argv = [sys.argv[0], sys.argv[1], "true", "true", "true", "true", "true",
folder_name + "/count_log.csv",
folder_name + "/countmodel_" + str(start_restore_index) + ".ckpt",
folder_name + "/countmodel_",
"true", "false", "false", "true"] #sys.argv[10]~[13]
print(sys.argv)

train_iters = 500000#20000000000
eps = 1e-8 # epsilon for numerical stability
rigid_pretrain = True
log_filename = sys.argv[7]
settings_filename = folder_name + "/settings.txt"
load_file = sys.argv[8]
save_file = sys.argv[9]
classify = str2bool(sys.argv[10]) #True
translated = str2bool(sys.argv[11]) #False
dims = [img_height, img_width]
img_size = dims[1]*dims[0] # canvas size
read_n = 15  # N x N attention window
read_size = read_n*read_n
output_size = max_blobs_train - min_blobs_train + 1
h_point_size = 256
h_count_size = 256
restore = str2bool(sys.argv[12]) #False
start_non_restored_from_random = str2bool(sys.argv[13]) #True
# delta, sigma2
delta_1=max(dims[0],dims[1])*1.5/(read_n-1)
sigma2_1=delta_1*delta_1/4 # sigma=delta/2 
delta_2=max(dims[0],dims[1])/3/(read_n-1)
sigma2_2=delta_2*delta_2/4 # sigma=delta/2

## BUILD MODEL ## 

REUSE = None

x = tf.placeholder(tf.float32,shape=(batch_size, img_size)) # input img
testing = tf.placeholder(tf.bool) # testing state
onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, output_size))
blob_list = tf.placeholder(tf.float32, shape=(batch_size, glimpses, 2))
size_list = tf.placeholder(tf.float32, shape=(batch_size, glimpses))
mask_list = tf.placeholder(tf.float32, shape=(batch_size, glimpses))
num_list = tf.placeholder(tf.float32, shape=(batch_size))
count_word = tf.placeholder(tf.float32, shape=(batch_size, glimpses, output_size + 1)) # add "I'm done!" signal
lstm_point = tf.contrib.rnn.LSTMCell(h_point_size, state_is_tuple=True) # point OP 
lstm_count = tf.contrib.rnn.LSTMCell(h_count_size, state_is_tuple=True) # count OP 

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    #JLM: small initial weights instead of N(0,1)
    w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=tf.random_uniform_initializer(minval=-.1, maxval=.1))
    #w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=tf.random_normal_initializer())
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def add_pointer(x, gx, gy, N):
    gx_idx = tf.reshape(tf.cast(gx, tf.int32), [1])
    gy_idx = tf.reshape(tf.cast(gy, tf.int32), [1])

    i_idx = (gy_idx-int(p_size/2))*img_width + (gx_idx-int(p_size/2))
    ii_idx = p_size
    iii_idx = (gx_idx-int(p_size/2)) + img_width - (gx_idx-int(p_size/2)) - p_size
    iv_idx = img_width - (gx_idx-int(p_size/2)) - int(p_size) + img_width*(img_height - ((gy_idx-int(p_size/2)) + int(p_size)))

    # constrain min_idx 
    min_idx = np.array([0])
    tmin_idx = tf.convert_to_tensor(min_idx, dtype=tf.int32)
    #i_idx = tf.maximum(i_idx, tmin_idx)
    #iv_idx = tf.maximum(iv_idx, tmin_idx)

    i = tf.ones(i_idx)
    ii = tf.ones(ii_idx)*255 # pointer blob
    iii = tf.ones(iii_idx)
    iv = tf.ones(iv_idx)

    pointer = tf.concat([tf.concat([tf.concat([tf.concat([i,ii], 0), iii], 0), ii], 0), iv], 0)
    pointer = tf.reshape(pointer, [1,img_width*img_height])
    x_pointer = x * pointer

    def cond(x_pointer):
        maxval = tf.ones(1, tf.float32)*255
        return tf.less(x_pointer[0, tf.argmax(x_pointer, 1)[0]], maxval)[0]

    def body(x_pointer):
        idx = tf.cast(tf.argmax(x_pointer, 1)[0], tf.int32)
        xx = tf.concat([x_pointer[0,0:idx], tf.ones(1)*255], 0)
        xxx = tf.concat([xx, x_pointer[0,idx+1:img_width*img_height]], 0)
        x_pointer = tf.reshape(xxx, [batch_size, img_width*img_height])
        return x_pointer

    x_pointer = tf.while_loop(cond, body, [x_pointer])

    return x_pointer

def filterbank(gx, gy, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x_1 = gx + (grid_i - N / 2 + 0.5) * delta_1 # eq 19 batch_size x N
    mu_y_1 = gy + (grid_i - N / 2 + 0.5) * delta_1 # eq 20 batch_size x N
    mu_x_2 = gx + (grid_i - N / 2 + 0.5) * delta_2
    mu_y_2 = gy + (grid_i - N / 2 + 0.5) * delta_2
    a = tf.reshape(tf.cast(tf.range(dims[0]), tf.float32), [1, 1, -1]) # 1 x 1 x dims[0]
    b = tf.reshape(tf.cast(tf.range(dims[1]), tf.float32), [1, 1, -1]) # 1 x 1 x dims[1]

    mu_x_1 = tf.reshape(mu_x_1, [-1, N, 1]) # batch_size x N x 1
    mu_y_1 = tf.reshape(mu_y_1, [-1, N, 1])
    mu_x_2 = tf.reshape(mu_x_2, [-1, N, 1])
    mu_y_2 = tf.reshape(mu_y_2, [-1, N, 1])
    Fx_1 = tf.exp(-tf.square(a - mu_x_1) / (2*sigma2_1)) # batch_size x N x dims[0]
    Fy_1 = tf.exp(-tf.square(b - mu_y_1) / (2*sigma2_1)) # batch_size x N x dims[1]
    Fx_2 = tf.exp(-tf.square(a - mu_x_2) / (2*sigma2_2)) # batch_size x N x dims[0]
    Fy_2 = tf.exp(-tf.square(b - mu_y_2) / (2*sigma2_2)) # batch_size x N x dims[1]
    # normalize, sum over A and B dims
    Fx_1=Fx_1/tf.maximum(tf.reduce_sum(Fx_1,2,keep_dims=True),eps)
    Fy_1=Fy_1/tf.maximum(tf.reduce_sum(Fy_1,2,keep_dims=True),eps)
        Fx_2=Fx_2/tf.maximum(tf.reduce_sum(Fx_2,2,keep_dims=True),eps)
    Fy_2=Fy_2/tf.maximum(tf.reduce_sum(Fy_2,2,keep_dims=True),eps)
    return Fx_1,Fy_1,Fx_2,Fy_2

def attn_window(scope, blob_list, h_point, N, glimpse, gx_prev, gy_prev, testing):
    with tf.variable_scope(scope,reuse=REUSE):
        params=linear(h_point,2) # batch_size x 2 
    gx_,gy_=tf.split(params, 2, 1) # batch_size x 1

    if glimpse==-1:
        gx_ = tf.zeros([batch_size,1])
        gy_ = tf.zeros([batch_size,1])
        glimpse = 0

    # relative distance
    gx_real = gx_prev + gx_
    gy_real = gy_prev + gy_

    # constrain gx and gy
    max_gx = np.array([dims[0]-1])
    tmax_gx = tf.convert_to_tensor(max_gx, dtype=tf.float32)
    gx_real = tf.minimum(gx_real, tmax_gx)

    min_gx = np.array([0])
    tmin_gx = tf.convert_to_tensor(min_gx, dtype=tf.float32)
    gx_real = tf.maximum(gx_real, tmin_gx)

    max_gy = np.array([dims[1]-1])
    tmax_gy = tf.convert_to_tensor(max_gy, dtype=tf.float32)
    gy_real = tf.minimum(gy_real, tmax_gy)

    min_gy = np.array([0])
    tmin_gy = tf.convert_to_tensor(min_gy, dtype=tf.float32)
    gy_real = tf.maximum(gy_real, tmin_gy)

    gx = tf.cond(testing, lambda:gx_real, lambda:tf.ones((batch_size, 1))*blob_list[0][glimpse][0])
    gy = tf.cond(testing, lambda:gy_real, lambda:tf.ones((batch_size, 1))*blob_list[0][glimpse][1])

    gx_prev = gx
    gy_prev = gy

    Fx_1, Fy_1, Fx_2, Fy_2 = filterbank(gx, gy, N)
    return Fx_1, Fy_1, Fx_2, Fy_2, gx, gy, gx_real, gy_real

## READ ## 
def read(x, h_point_prev, glimpse, testing):
    Fx_1, Fy_1, Fx_2, Fy_2, gx, gy, gx_real, gy_real = attn_window("read", blob_list, h_point_prev, read_n, glimpse, gx_prev, gy_prev, testing)
    stats = Fx_1, Fy_1, Fx_2, Fy_2
    new_stats = gx, gy, gx_real, gy_real
    # x = add_pointer(x, gx, gy, read_n)

    def filter_img(img, Fx_1, Fy_1, Fx_2, Fy_2, N):
        Fxt_1 = tf.transpose(Fx_1, perm=[0,2,1])
        Fxt_2 = tf.transpose(Fx_2, perm=[0,2,1])
        # img: 1 x img_size
        img = tf.reshape(img,[-1, dims[1], dims[0]])
        fimg_1 = tf.matmul(Fy_1, tf.matmul(img, Fxt_1))
        fimg_1 = tf.reshape(fimg_1,[-1, N*N])
        fimg_2 = tf.matmul(Fy_2, tf.matmul(img, Fxt_2))
                fimg_2 = tf.reshape(fimg_2,[-1, N*N])
        # normalization
        fimg_1 = fimg_1/tf.reduce_max(fimg_1, 1, keep_dims=True)
        fimg_2 = fimg_2/tf.reduce_max(fimg_2, 1, keep_dims=True)
        fimg = tf.concat([fimg_1, fimg_2], 1)
        return tf.reshape(fimg, [batch_size, -1])

    xr = filter_img(x, Fx_1, Fy_1, Fx_2, Fy_2, read_n) # batch_size x (read_n*read_n)
    return xr, new_stats # concat along feature axis

## POINTER ##
def pointer(input, state):
    """
    run LSTM
    state: previous lstm_cell state
    input: cat(read, h_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("point/LSTMCell", reuse=REUSE):
        return lstm_point(input, state)

## COUNTER ##
def counter(input, state):
    """
    run LSTM
    state: previous lstm_cell state
    input: cat(read, h_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("count/LSTMCell", reuse=REUSE):
        return lstm_count(input, state)


## STATE VARIABLES ##############
# initial states
gx_prev = tf.zeros((batch_size, 1))
gy_prev = tf.ones((batch_size, 1))*dims[1]/2
h_point_prev = tf.zeros((batch_size, h_point_size))
h_count_prev = tf.zeros((batch_size, h_count_size))
point_state = lstm_point.zero_state(batch_size, tf.float32)
count_state = lstm_count.zero_state(batch_size, tf.float32)
