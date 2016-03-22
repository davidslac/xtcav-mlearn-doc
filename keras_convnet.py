from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from glob import glob
import time
import random
import numpy as np
import scipy.stats

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1l2, activity_l1, activity_l2, activity_l1l2
from keras.optimizers import SGD
from keras import backend as K

def build_model(model):
    regularize=0.01
    ## layer 1 -- input is about 600 x 800
    ch01 = 16
    xx01 = 12
    yy01 = 12
    kern_W_init = (0.06/2.0)*scipy.stats.truncnorm.rvs(-2.0, 2.0, size=(ch01, 1, xx01, yy01)).astype(np.float32)
    kern_B_init = np.zeros(ch01, dtype=np.float32)
    model.add(Convolution2D(ch01, xx01, yy01, border_mode='same', 
                            weights=[kern_W_init, kern_B_init],
                            W_regularizer=l2(regularize),
                            input_shape=datareader.features_placeholder_shape()[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3))) #, strides=(4,4)))
    model.add(BatchNormalization(epsilon=1e-06, 
                                 mode=0, 
                                 axis=1, momentum=0.9, weights=None, 
                                 beta_init='zero', gamma_init='one'))

    ## layer 2  input is about 200 x 270
    ch02 = 16
    xx02 = 10
    yy02 = 10
    kern_W_init = (0.06/2.0)*scipy.stats.truncnorm.rvs(-2.0, 2.0, size=(ch02, ch01, xx02, yy02)).astype(np.float32)
    kern_B_init = np.zeros(ch02, dtype=np.float32)
    model.add(Convolution2D(ch02, xx02, yy02, 
                            border_mode='same', 
                            W_regularizer=l2(regularize),
                            weights=[kern_W_init, kern_B_init]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5,5)))#, strides=(4,4)))
    model.add(BatchNormalization(epsilon=1e-06, 
                                 mode=0, 
                                 axis=1, momentum=0.9, weights=None, 
                                 beta_init='zero', gamma_init='one'))
    
    ## layer 3 input is about 50 x 60
    ch03 = 16
    xx03 = 8
    yy03 = 8
    kern_W_init = (0.06/2.0)*scipy.stats.truncnorm.rvs(-2.0, 2.0, size=(ch03, ch02, xx03, yy03)).astype(np.float32)
    kern_B_init = np.zeros(ch03,dtype=np.float32)
    model.add(Convolution2D(ch03, xx03, yy03, 
                            border_mode='same', 
                            W_regularizer=l2(regularize),
                            weights=[kern_W_init, kern_B_init]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5,5)))#, strides=(4,4)))
    model.add(BatchNormalization(epsilon=1e-06, 
                                 mode=0, 
                                 axis=1, momentum=0.9, weights=None, 
                                 beta_init='zero', gamma_init='one'))
    
    
    model.add(Flatten())
    
    ## layer 4  input is about 12 x 15
    hidden04 = 48
    W_init =  (0.06/2.0)*scipy.stats.truncnorm.rvs(-2.0, 2.0, size=(1008,hidden04)).astype(np.float32)
    B_init = np.zeros(hidden04, dtype=np.float32)
    model.add(Dense(hidden04, 
                    W_regularizer=l2(regularize),
                    weights=[W_init, B_init]))
    model.add(Activation('relu'))
    model.add(BatchNormalization(epsilon=1e-06, 
                                 mode=0, 
                                 axis=1, momentum=0.9, weights=None, 
                                 beta_init='zero', gamma_init='one'))
  
    ## layer 5
    hidden05 = 32
    W_init =  (0.06/2.0)*scipy.stats.truncnorm.rvs(-2.0, 2.0, size=(hidden04, hidden05)).astype(np.float32)
    B_init = np.zeros(hidden05,dtype=np.float32)
    model.add(Dense(hidden05, 
                    W_regularizer=l2(regularize),
                    weights=[W_init, B_init]))
    model.add(Activation('relu'))
    model.add(BatchNormalization(epsilon=1e-06, 
                                 mode=0, 
                                 axis=1, momentum=0.9, weights=None, 
                                 beta_init='zero', gamma_init='one'))

    ## layer 6
    W_init =  (0.06/2.0)*scipy.stats.truncnorm.rvs(-2.0, 2.0, size=(hidden05, datareader.num_outputs())).astype(np.float32)
    B_init = np.zeros(datareader.num_outputs(),dtype=np.float32)
    model.add(Dense(datareader.num_outputs(), 
                    weights=[W_init, B_init],
                    W_regularizer=l2(0.2*regularize)))
    model.add(Activation('softmax'))

def set_learning_rate(model, step_number):
    min_learning_rate = 0.001
    steps_per_drop = 100
    drop_by = 0.98

    lr = model.optimizer.lr.get_value()
                                                      
    if lr <= min_learning_rate:
        return lr
    if step_number % steps_per_drop == (steps_per_drop-1):
        lr *= drop_by

    K.set_value(model.optimizer.lr, lr)
    return lr

def get_confusion_matrix_one_hot(model_results, truth):
    assert model_results.shape == truth.shape
    num_outputs = truth.shape[1]
    cmat = np.zeros((num_outputs, num_outputs), dtype=np.int32)
    predictions = np.argmax(model_results,axis=1)
    assert len(predictions)==truth.shape[0]

    for actual_class in range(num_outputs):
        idx_examples_this_class = truth[:,actual_class]==1
        prediction_for_this_class = predictions[idx_examples_this_class]
        for predicted_class in range(num_outputs):
            count = np.sum(prediction_for_this_class==predicted_class)
            cmat[actual_class, predicted_class] = count
    assert np.sum(cmat)==len(truth)
    return cmat

def eval_report(epoch, batch, step_times, lr, train_loss, train_accuracy, model, train_features, train_labels, validation_features, validation_labels):
    t0 = time.time()
    sec_per_step = np.sum(step_times)/len(step_times)

    validation_predict = model.predict(validation_features)
    confusion_matrix_validation = get_confusion_matrix_one_hot(validation_predict, validation_labels)
    eval_accuracy = np.trace(confusion_matrix_validation)/np.sum(confusion_matrix_validation)

    train_predict = model.predict(train_features)
    confusion_matrix_train = get_confusion_matrix_one_hot(train_predict, train_labels)
    train_accuracy_from_confusion_matrix = np.trace(confusion_matrix_train)/np.sum(confusion_matrix_train)

    def fmt3(x): return '%3d' % x
    cmat_tr_row0 = ' '.join(map(fmt3, confusion_matrix_train[0,:]))
    cmat_ev_row0 = ' '.join(map(fmt3, confusion_matrix_validation[0,:]))
    eval_time = time.time()-t0
    print(" %2d:%4d |%5.1f |%8.2f |%7.2f/%6.2f |%7.2f |%8.5f | %s | %s | %5.1f" %
          (epoch, batch, sec_per_step,
           train_loss, train_accuracy, 
           train_accuracy_from_confusion_matrix,
           eval_accuracy, 
           lr, cmat_tr_row0, cmat_ev_row0, eval_time))
    for row in range(1,confusion_matrix_train.shape[0]):
        print(" %s | %s | %s |" % (' '*59, 
                                    ' '.join(map(fmt3, confusion_matrix_train[row,:])),
                                    ' '.join(map(fmt3, confusion_matrix_validation[row,:]))))
    print("-"*99)

def run(datareader):
    start_time = time.time()
    batches_per_epoch =  datareader.batches_per_epoch()
    print("keras_convnet: datareader - %d batches per epoch" % batches_per_epoch)
    validation_features, validation_labels = datareader.get_validation_set()

    print("starting to build and compile keras/theano model...")
    sys.stdout.flush()
    t0 = time.time()

    model = Sequential()
    build_model(model)
    model.load_weights("keras_convnet_6layer.h5")

    sgd = SGD(lr=0.01, momentum=0.92)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    print("building/compiling theano model took %.2f sec" % (time.time()-t0),)
    sys.stdout.flush()

    print(" epch:mb | tm.s | loss.tr | acc.tr/ batch | acc.ev | learn.rt| confuse mat tr | confuse mat eva  | tm.e")
    eval_step_interval = 100
    num_steps = 8001
    step_times = []
    for step_number in range(1,num_steps+1):
        epoch = step_number // batches_per_epoch
        batch = step_number % batches_per_epoch
        t0 = time.time()
        train_features, train_labels = datareader.get_next_minibatch()
        lr = set_learning_rate(model, step_number)
        train_loss, train_accuracy = model.train_on_batch(train_features, train_labels, accuracy=True)
        step_time = time.time()-t0
        step_times.append(step_time)
        if step_number % eval_step_interval == 0:
            eval_report(epoch, batch, step_times, lr, train_loss, train_accuracy, model, train_features, train_labels, validation_features, validation_labels)
            step_times = []

#    print("model has %d params" % model.count_params())
    model.summary()
    model.save_weights("keras_convnet_6layer_B.h5", overwrite=True)
    print("total time: %.2f" % (time.time()-start_time))


assert datareader is not None, 'datareader is not defined' # in globals() nor locals()'
run(datareader)
