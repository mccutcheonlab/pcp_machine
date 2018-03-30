# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:40:03 2018

@author: jaimeHP
"""

import random
import pandas as pd

def slice_sessions(nsessions, trainN=120):
    
    train_s = random.sample(range(nsessions),trainN)
    test_s = [x for x in range(nsessions) if x not in train_s]
    return train_s, test_s

def assemble_data(sessionNs):
    
    df = pd.DataFrame([x for x in sessionNs])
    
    df.insert(1,'nlicks', [len(sessions[x].licks) for x in sessionNs])
    df.insert(2,'freq', [sessions[x].lickdata['freq'] for x in sessionNs])
    df.insert(3,'bNum', [sessions[x].lickdata['bNum'] for x in sessionNs])
    df.insert(3,'bMean', [sessions[x].lickdata['bMean'] for x in sessionNs])
    df.insert(4,'group', [sessions[x].group for x in sessionNs])

    return df
    
# load session data from pickled file
try:
    type(sessions)
    print('Using existing data')
except NameError:
    print('Loading in pickled file')
    try:
        pickle_in = open('C:\\Users\\jaimeHP\\Documents\\pcp_sessions.pickle', 'rb')
        rats = dill.load(pickle_in)
    except:
        print('Unable to find pickled file')
        

train_s, test_s = slice_sessions(len(sessions))

df_train = assemble_data(train_s)
df_test = assemble_data(test_s)

    