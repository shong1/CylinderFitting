# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 11:31:45 2015

@author: root
"""

import numpy as np
import scipy as sci
import scipy.linalg

def ransac_cylinder( pts, model, nHypoData, max_iter, hypoFit_thres, moodel_thres, d ):
	"""
	Input
	data - Data Points : Each row is a point (or a single datum)
	model - 4 * 4 Cylinder Matrix and its fitting, error function
	nHypoData - The number of required data points to estimate the model
	max_iter - The maximum iteration
	hypoFit_thres - The threshold of the individual fitting function with the selected hypothetical points
	model_thres - The threshold of the determined fitting function for all data
	"""

    ptsDim = pts.shape;
    iteration = 0
    best_hypoFitModel = None
    best_hypoErr = numpy.inf
    best_inlier_pts = None
    
    
    while iteration < max_iter:
        hypo_pts_idx = random_integers( 0, ptsDim[0], nHypoData )
        hypo_pts = data[ hypo_pts_idx ]      

class CylinderModel:
    """linear system solved using linear least squares
    This class serves as an example that fulfills the model interface
    needed by the ransac() function.
    """
    
    def __init__(self, pt_dim, cylinderMat,debug=False):
        self.pt_dim = pt_dim
        self.cylindreMat = cylinderMat
        self.debug = debug
    def fit(self, data):
        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
        x,resids,rank,s = scipy.linalg.lstsq(A,B)
        return x
    def get_error( self, data, model):
        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
        B_fit = scipy.dot(A,model)
        err_per_point = numpy.sum((B-B_fit)**2,axis=1) # sum squared error per row
        return err_per_point
        