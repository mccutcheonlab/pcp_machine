# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:40:03 2018

@author: jaimeHP
"""

import random
import pandas as pd
import dill

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

try:
    import tensorflow as tf
except:
    print('Tensor flow not available - run program from tf environment')
    
def slice_sessions(nsessions, trainN=120):
    
    train_s = random.sample(range(nsessions),trainN)
    test_s = [x for x in range(nsessions) if x not in train_s]
    return train_s, test_s

def assemble_data(sessionNs):
    
    df = pd.DataFrame([x for x in sessionNs])
    
    df.insert(1,'nlicks', [len(sessions[x].licks) for x in sessionNs])
    df.insert(2,'freq', [sessions[x].lickdata['freq'] for x in sessionNs])
    df.insert(3,'bNum', [sessions[x].lickdata['bNum'] for x in sessionNs])
    df.insert(4,'bMean', [sessions[x].lickdata['bMean'] for x in sessionNs])
    df.insert(5,'group', [sessions[x].group_numeric for x in sessionNs])
    
    return df

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

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
        
def main(argv):
    args = parser.parse_args(argv[1:])
    train_s, test_s = slice_sessions(len(sessions))
    
    train = assemble_data(train_s)
    test = assemble_data(test_s)
    
    (train_x, train_y) = train, train.pop('group')
    (test_x, test_y) = test, test.pop('group')
    
    my_feature_columns = []
    for key in train_x.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=2)
    
    print(train_x[:5])
    
#    classifier.train(
#        input_fn=lambda:train_input_fn(train_x, train_y,
#                                                 args.batch_size),
#        steps=args.train_steps)
        
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

    