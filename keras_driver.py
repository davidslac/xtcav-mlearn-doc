from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import traceback
from glob import glob
import time
from h5minibatch.H5MiniBatchReader import H5MiniBatchReader

assert len(sys.argv)==2, "provide the script to run"
keras_driver_script = sys.argv[1]
assert os.path.exists(keras_driver_script), "script %s doesn't exist" % keras_driver_script

keras_driver_datadir = "/reg/d/ana01/temp/davidsch/ImgMLearnFull"
keras_driver_h5files = glob(os.path.join(keras_driver_datadir, "amo86815_mlearn-r070*.h5"))
keras_driver_h5files.extend(glob(os.path.join(keras_driver_datadir, "amo86815_mlearn-r071*.h5")))
#keras_driver_h5files = ["/reg/d/ana01/temp/davidsch/ImgMLearnFull/amo86815_mlearn-r071-c0000.h5",
#                        "/reg/d/ana01/temp/davidsch/ImgMLearnFull/amo86815_mlearn-r071-c0001.h5",
#                        "/reg/d/ana01/temp/davidsch/ImgMLearnFull/amo86815_mlearn-r071-c0002.h5"]
assert len(keras_driver_h5files)>0
    
datareader = H5MiniBatchReader(h5files=keras_driver_h5files,
                               minibatch_size=64,
                               validation_size=400,
                               feature_dataset='xtcavimg',
                               label_dataset='acq.peaksLabel',
                               return_as_one_hot=True,
                               feature_preprocess=['log','mean'],
                               number_of_batches=None,
                               class_labels_max_imbalance_ratio=4.0,
                               add_channel_to_2D='channel_row_column',
                               max_mb_to_preload_all='all',
                               cache_preprocess=True,
                               random_seed=None, #23432,
                               verbose=True)  
    
while True:
#        scriptGlobals = {}
#        scriptLocals = {}
#        for key,value in globals().iteritems():
#            scriptGlobals[key]=value            
#        for key, value in locals().iteritems():
#            scriptLocals[key]=value
#        scriptGlobals['datareader'] = datareader
#        scriptLocals['datareader'] = datareader
#        scriptGlobals['__name__'] = os.path.splitext(os.path.basename(__file__))[0] # '__main__'
#        scriptGlobals['__file__'] = os.path.abspath(script)
    try:
        execfile(keras_driver_script) #, locals=scriptLocals)
    except Exception,e:
        print("Exeception: %s" % e)
        print('-'*60)
        traceback.print_exc(file=sys.stdout)
        print('-'*60)
        sys.stdout.flush()

    result = raw_input("type q to quit, or anything else to rerun script %s" % keras_driver_script).strip().lower()
    if result == 'q' or result == 'quit':
        print("quit received")
        sys.exit(0)
        
    
