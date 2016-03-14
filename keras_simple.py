from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from glob import glob
import time

import numpy as np
import scipy.stats
from h5minibatch.H5MiniBatchReader import H5MiniBatchReader

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

def run():
    datadir = "/reg/d/ana01/temp/davidsch/ImgMLearnFull"
    h5files = glob(os.path.join(datadir, "amo86815_mlearn-r070*.h5"))
    h5files.extend(glob(os.path.join(datadir, "amo86815_mlearn-r071*.h5")))
#    h5files = ["/reg/d/ana01/temp/davidsch/ImgMLearnFull/amo86815_mlearn-r071-c0000.h5",
#               "/reg/d/ana01/temp/davidsch/ImgMLearnFull/amo86815_mlearn-r071-c0001.h5",
#               "/reg/d/ana01/temp/davidsch/ImgMLearnFull/amo86815_mlearn-r071-c0002.h5"]
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
                                   add_channel_to_2D='channel_row_column',
                                   max_mb_to_preload_all=None,
                                   random_seed=None,
                                   verbose=True)  

    validation_features, validation_labels = datareader.get_validation_set()

    print("starting to build and compile keras/theano model...")
    sys.stdout.flush()
    t0 = time.time()
    model = Sequential()

    ## layer 1
    kern01_W_init = (0.06/2.0)*scipy.stats.truncnorm.rvs(-2.0, 2.0, size=(8,1,8,8)).astype(np.float32)
    kern01_B_init = np.zeros(8,dtype=np.float32)
    model.add(Convolution2D(8,8,8, border_mode='same', weights=[kern01_W_init, kern01_B_init],
                            input_shape=datareader.features_placeholder_shape()[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(10,10), strides=(13,13)))
    
    ## layer 2
    kern02_W_init = (0.06/2.0)*scipy.stats.truncnorm.rvs(-2.0, 2.0, size=(8,8,6,6)).astype(np.float32)
    kern02_B_init = np.zeros(8,dtype=np.float32)
    model.add(Convolution2D(8,6,6, border_mode='same', weights=[kern02_W_init, kern02_B_init]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(10,10), strides=(13,13)))
    
    model.add(Flatten())
    
    ## layer 3
    H03_W_init =  (0.06/2.0)*scipy.stats.truncnorm.rvs(-2.0, 2.0, size=(96,16)).astype(np.float32)
    H03_B_init = np.zeros(16,dtype=np.float32)
    model.add(Dense(16, weights=[H03_W_init, H03_B_init]))
    model.add(Activation('relu'))
    
    ## layer 4
    H04_W_init =  (0.06/2.0)*scipy.stats.truncnorm.rvs(-2.0, 2.0, size=(16, datareader.num_outputs())).astype(np.float32)
    H04_B_init = np.zeros(datareader.num_outputs(),dtype=np.float32)
    model.add(Dense(datareader.num_outputs(), weights=[H04_W_init, H04_B_init]))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=0.0004, momentum=0.96)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    print("building/compiling theano model took %.2f sec" % (time.time()-t0),)
    sys.stdout.flush()

    for step_number in range(3000):
        t0 = time.time()
        train_features, train_labels = datareader.get_next_minibatch()
        model.train_on_batch(train_features, train_labels)
        print("step %3d took %.2f sec." % (step_number, time.time()-t0))
        sys.stdout.flush()

    print("Starting evaluation.")
    t0 = time.time()
    loss, validation_accuracy = model.test_on_batch(validation_features, validation_labels, accuracy=True, sample_weight=None)
    print("validation accuracy: %.2f%%" % (100.0*validation_accuracy,))
    print("evaluation took %.2f sec" % (time.time()-t0,))
    
    
if __name__ == '__main__':
    run()
    
