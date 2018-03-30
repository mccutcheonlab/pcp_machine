# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 13:10:45 2018

@author: jaimeHP
"""

import dill

pickle_out = open('C:\\Users\\jaimeHP\\Documents\\pcp_sessions.pickle', 'wb')
dill.dump(sessions, pickle_out)
pickle_out.close()