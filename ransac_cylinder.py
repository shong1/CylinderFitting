# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 11:31:45 2015

@author: root
"""

import numpy as np
import scipy as sci
import scipy.linalg

def ransac_cylinder(data,model,n,k,t,d,debug=False,return_all=False):

"""
Input
data - 