# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:35:48 2018

@author: jaimeHP
"""
import numpy as np

import JM_general_functions as jmf

class Session(object):
    
    def __init__(self, rowdata):
        self.rowdata = rowdata
        self.medfile = folder + self.rowdata[0]
        self.licks = jmf.medfilereader(self.medfile, varsToExtract = ['e'],
                                       remove_var_header = True)
    
    def assign_group(self):
        self.group = self.rowdata[5]
        
    def lick_analysis(self):
        self.lickdata = jmf.lickCalc(self.licks)

def strip_commas(data_in):
    data_out = []
    for i in data_in:
        data_out.append(i[0].split(','))
        
    return data_out

folder = 'R:\\DA_and_Reward\\kp259\\DPCP1\\'
metafile = folder + 'DPCP1Masterfile.csv'

data, headers = jmf.metafilereader(metafile)
data = strip_commas(data)

try:
    type(sessions)
    print('Using existing data')
except NameError:
    sessions = {}
    for i,row in enumerate(data):
        sessions[i] = Session(row)
    
for i in sessions:
    x = sessions[i]
    x.assign_group()
    x.lick_analysis()

