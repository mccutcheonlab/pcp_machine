# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 07:37:36 2018

@author: jaimeHP
"""

from skimage import transform
import dill
import numpy as np
import random
import pandas as pd

# load session data from pickled file
try:
    type(sessions)
    print('Using existing data')
except NameError:
    print('Trying to load pickled file')
    try:
        pickle_in = open('C:\\Users\\jaimeHP\\Documents\\pcp_sessions.pickle', 'rb')
        sessions = dill.load(pickle_in)
        print('File loaded successfully')
    except:
        print('Unable to load or find pickled file. Make sure dill is imported and file location is correct')

def resize_licks(licks):
    try:
        licks = np.reshape(licks,(-1,2))
    except:
        licks = np.reshape(licks[:-1],(-1,2))
    
    licks_resized = transform.resize(licks, (28,28))

    return licks_resized

def slice_sessions(nsessions, trainN=120):   
    train_s = random.sample(range(nsessions),trainN)
    test_s = [x for x in range(nsessions) if x not in train_s]
    return train_s, test_s

def load_data(datamode='all', datatype='ilis'):      
    data = []
    if datamode == 'limited':
        for i in range(len(sessions)):
            session = [x for x in sessions[i].lickdata[datatype]][:784]
            print(len(session))
            data.append(resize_licks(session))        
    else:
        for i in range(len(sessions)):
            data.append(resize_licks(sessions[i].lickdata[datatype]))
    
    train_s, test_s = slice_sessions(len(sessions))

    train_x = np.array([data[x] for x in train_s], dtype=np.float32)
    test_x = np.array([data[x] for x in test_s], dtype=np.float32)

    train_y = np.asarray(pd.DataFrame(
            [sessions[x].group_numeric for x in train_s],
            columns=['group']), dtype=np.int32)[:,0]
    
    test_y = np.asarray(pd.DataFrame(
        [sessions[x].group_numeric for x in test_s],
        columns=['group']), dtype=np.int32)[:,0]
    
    return (train_x, train_y), (test_x, test_y)

def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

#(train_x, train_y), (test_x, test_y) = load_data(datamode='limited')

