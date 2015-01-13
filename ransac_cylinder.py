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
	best_hypoErr = np.inf
	best_inlier_pts = None
        
	while iteration < max_iter:
		hypo_pts_idx = random_integers( 0, ptsDim[0], nHypoData )
		hypo_pts = data[ hypo_pts_idx ]
		

class CylinderModel:
"""linear system solved using linear least squares
This class serves as an example that fulfills the model interface
needed by the ransac() function.
"""
	def __init__(self, debug=False):
		self.debug = debug
	def fit(self, data):
		dataDim = data.shape
		meanData = np.mean( data, axis = 0 )
		data = np.subtract( data, meanData )
		minFitErr = np.inf
		W = np.zeros( ( 1, 3 ) )
		C = np.zeros( ( 1, 3 ) )
		rSqr = 0
		
		for phi in xrange( 0, pi / 2, 0.1 ):
			cPhi = np.cos( phi )
			sPhi = np.sin( phi )
			for theta in xrange( 0, 2 * pi, 0.1 ):
				cTheta = np.cos( theta )
				sTheta = np.sin( theta )
				tW = np.array( [ cTheta * sPhi, sTheta * sPhi, cPhi ] )
				tC = np.zeros( ( 1, 3 ) ) 
				tErr, tRSqr = get_error( data, tW, tC )

		return x
	def get_error( self, data, W, C ):
		dataDim = data.shape
		
		S = np.array( [ [ 0, -W[2], W[1] ], [ W[2], 0, -W[0] ], [ -W[1], W[0], 0 ] ] )		
		P = np.subtract( np.identity( 3 ), np.outer( W, W ) )
		A = np.zeros( ( 3, 3 ) )
		B = np.zeros( ( 1, 3 ) )
		mSqrLength = 0
		SqrLength = np.zeros( ( dataDim[0], 1 ) )
		Y = np. zeros( ( dataDim[0], 3 ) )
		
		for i in xrange( dataDim[0] ):
			Yi = np.inner( P, data[ i, : ] )
			sqrLength[ i ] = np.inner( Yi, Yi )
			A += np.add( np.outer( Yi, Yi ) )
			B += sqrLength[ i ] * Yi
			mSqrLength = sqrLength[ i ]
			Y[ i, : ] = Yi
		
		A = np.divide( A, dataDim[ 0 ] )
		B = np.divide( B, dataDim[ 0 ] )
		mSqrLength = mSqrLength / dataDim[ 0 ]
		A_hat = np.inner( -S, np.inner( A, S ) )
		PC = 
		
	 	
		return err, rSqr