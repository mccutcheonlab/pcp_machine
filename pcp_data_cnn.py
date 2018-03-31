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

def resize_licks(licks, flatten=False):
    try:
        licks = np.reshape(licks,(-1,2))
    except:
        licks = np.reshape(licks[:-1],(-1,2))
    
    licks_resized = transform.resize(licks, (28,28))
    
#    if flatten == True:
#        for px in licks_resized:
#            
#        licks_resized = [mean(x)]
    return licks_resized

def slice_sessions(nsessions, trainN=120):   
    train_s = random.sample(range(nsessions),trainN)
    test_s = [x for x in range(nsessions) if x not in train_s]
    return train_s, test_s

def load_data():
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

data = []
for i in range(len(sessions)):
    data.append(resize_licks(sessions[i].lickdata['ilis']))

#test = resize_licks(sessions[0].licks)
#print(test[1,2])
    
# Idea - go through sessions, only keeping those with ilis>28x28 then rather than resizing, just change shape

