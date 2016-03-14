from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from glob import glob
import time
import numpy as np
from h5minibatch.H5MiniBatchReader import H5MiniBatchReader
import tensorflow as tf

def make_nnet_ops(features, labels, num_outputs):
    ## layer 1 
    kern01 = tf.Variable(tf.truncated_normal([8,8,1,8], mean=0.0, stddev=0.03))
    conv01 = tf.nn.conv2d(features, kern01, strides=(1,1,1,1), padding="SAME")
    bias01 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[8]))
    addBias01 = tf.nn.bias_add(conv01, bias01)
    nonlinear01 =tf.nn.relu(addBias01)
    pool01 = tf.nn.max_pool(value=nonlinear01, ksize=(1,13,13,1), 
                            strides=(1,10,10,1), padding="SAME")

    ## layer 2
    kern02 = tf.Variable(tf.truncated_normal([6,6,8,8], mean=0.0, stddev=0.03))
    conv02 = tf.nn.conv2d(pool01, kern02, strides=(1,1,1,1), padding="SAME")
    bias02 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[8]))
    addBias02 = tf.nn.bias_add(conv02, bias02)
    nonlinear02 =tf.nn.relu(addBias02)
    pool02 = tf.nn.max_pool(value=nonlinear02, ksize=(1,13,13,1), 
                            strides=(1,10,10,1), padding="SAME")
    
    num_inputs_to_layer03 = 1
    for dim in pool02.get_shape()[1:].as_list():
        num_inputs_to_layer03 *= dim
    input_to_layer03 = tf.reshape(pool02, [-1, num_inputs_to_layer03])

    ## layer 3
    weights03 = tf.Variable(tf.truncated_normal([num_inputs_to_layer03, 16], mean=0.0, stddev=0.03))
    bias03 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[16]))
    xw_plus_b = tf.nn.xw_plus_b(input_to_layer03, weights03, bias03)
    nonlinear03 = tf.nn.relu(xw_plus_b)

    ## layer 4
    weights04 = tf.Variable(tf.truncated_normal([16, num_outputs], mean=0.0, stddev=0.1))
    bias04 = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[num_outputs]))
    logits =  tf.nn.xw_plus_b(nonlinear03, weights04, bias04)

    ## loss 
    cross_entropy_loss_all = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss_all)

    ## training
    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(learning_rate=0.01,
                                               global_step=global_step,
                                               decay_steps=100,
                                               decay_rate=0.96,
                                               staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.85)
    train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)

    return logits, cross_entropy_loss, train_op

def run():
    datadir = "/reg/d/ana01/temp/davidsch/ImgMLearnFull"
    h5files = glob(os.path.join(datadir, "amo86815_mlearn-r070*.h5"))
    h5files.extend(glob(os.path.join(datadir, "amo86815_mlearn-r071*.h5")))
    assert len(h5files)>0
    datareader = H5MiniBatchReader(h5files=h5files,
                                   minibatch_size=32,
                                   validation_size=64,
                                   feature_dataset='xtcavimg',
                                   label_dataset='acq.peaksLabel',
                                   return_as_one_hot=True,
                                   feature_preprocess=['log','mean'],
                                   number_of_batches=None,
                                   class_labels_max_imbalance_ratio=1.0,
                                   max_mb_to_preload_all=None,
                                   add_channel_to_2D='row_column_channel',
                                   random_seed=None,
                                   verbose=True)    
    with tf.Graph().as_default():
        with_graph(datareader)

def with_graph(datareader):
    validation_features, validation_labels = datareader.get_validation_set()
    num_outputs = datareader.num_outputs()
    features_shape = datareader.features_placeholder_shape()
    features_tensor = tf.placeholder(tf.float32, shape=features_shape)
    labels_tensor = tf.placeholder(tf.float32, shape=(None, num_outputs))

    logits, cross_entropy_loss, train_op = make_nnet_ops(features_tensor, 
                                                         labels_tensor,
                                                         num_outputs)

    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads = 12))

    init = tf.initialize_all_variables()

    sess.run(init)
    
    validation_feed_dict = {features_tensor:validation_features,
                            labels_tensor:validation_labels}    
    
    print("Starting training.")
    sys.stdout.flush()
    for step_number in range(3000):
        t0 = time.time()
        train_features, train_labels = datareader.get_next_minibatch()
        train_feed_dict = {features_tensor:train_features,
                           labels_tensor:train_labels}
        sess.run(train_op, feed_dict=train_feed_dict)        
        print("step %3d took %.2f sec." % (step_number, time.time()-t0))
        sys.stdout.flush()
    
    print("Starting evaluation.")
    sys.stdout.flush()

    t0 = time.time()
    logits_validation = sess.run(logits, feed_dict=validation_feed_dict)

    validation_accuracy = np.sum(np.argmax(logits_validation,1) ==
                                 np.argmax(validation_labels,1))/len(validation_labels)

    print("validation accuracy: %.2f%%" % (100.0*validation_accuracy,))
    print("evaluation took %.2f sec" % (time.time()-t0,))

if __name__ == '__main__':
    run()
