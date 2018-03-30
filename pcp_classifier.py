# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:35:38 2018

@author: jaimeHP
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import pcp_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])
    
    (train_x, train_y), (test_x, test_y) = pcp_data.load_data()
    
    my_feature_columns = []
    for key in ['freq', 'bMean']:
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=2)

    classifier.train(
        input_fn=lambda:pcp_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)
        
        
#        # Evaluate the model.
#    eval_result = classifier.evaluate(
#        input_fn=lambda:pcp_data.eval_input_fn(test_x, test_y,
#                                                args.batch_size))
#
#    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)