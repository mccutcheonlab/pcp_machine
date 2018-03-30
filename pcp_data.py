# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:40:03 2018

@author: jaimeHP
"""

import random
import pandas as pd
import dill

import argparse

try:
    import tensorflow as tf
except:
    print('Tensor flow not available - run program from tf environment')
    
GROUP = ['SAL', 'PCP']    
CSV_COLUMN_NAMES = ['s', 'nlicks', 'freq', 'bNum', 'bMean', 'group']

def slice_sessions(nsessions, trainN=120):
    
    train_s = random.sample(range(nsessions),trainN)
    test_s = [x for x in range(nsessions) if x not in train_s]
    return train_s, test_s

def assemble_data(sessionNs):
    
    df = pd.DataFrame([len(sessions[x].licks) for x in sessionNs], columns=['nlicks'])
    df.insert(1,'freq', [sessions[x].lickdata['freq'] for x in sessionNs])
    df.insert(2,'bNum', [sessions[x].lickdata['bNum'] for x in sessionNs])
    df.insert(3,'bMean', [sessions[x].lickdata['bMean'] for x in sessionNs])
    df.insert(4,'group', [sessions[x].group_numeric for x in sessionNs])
    
    return df

def load_data(y_name='group'):
    train_s, test_s = slice_sessions(len(sessions))
    
    train = assemble_data(train_s)
    test = assemble_data(test_s)

    train_x, train_y = train, train.pop(y_name)
    test_x, test_y = test, test.pop(y_name)
    
    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

# load session data from pickled file
try:
    type(sessions)
    print('Using existing data')
except NameError:
    print('Loading in pickled file')
    try:
        pickle_in = open('C:\\Users\\jaimeHP\\Documents\\pcp_sessions.pickle', 'rb')
        sessions = dill.load(pickle_in)
    except:
        print('Unable to find pickled file')
        
        


    