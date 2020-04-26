# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import sys
from hyperparams import Hyperparams as hp
from data_load import load_source_data, load_vocab
from train import Graph
                           
# Load graph
g = Graph(is_training=False)
print("Graph loaded")
char2idx, idx2char = load_vocab()

# Start session         
with g.graph.as_default():    
    sv = tf.train.Supervisor()  #logdir=hp.logdir
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ## Restore parameters
        sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
        while True:
            sent = input("输入上联(quit)：")
            sent = sent.replace(' ','')
            ## exit 
            if sent.lower() == 'quit':
                break

            source = [char for char in sent.strip()]
            x = [char2idx.get(word, 1) for word in (source + ["</S>"])]
            # Pad      
            x = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
            x = np.reshape(x, (1, 30))
            ### Autoregressive inference
            preds = np.zeros((1, hp.maxlen), np.int32)
            for j in range(hp.maxlen):
                _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                preds[:, j] = _preds[:, j]

            ### Write to file
            got = "".join(idx2char[idx] for idx in preds[0]).split("</S>")[0].strip()
            print("- 我对下联: " + got + "\n")
    
    