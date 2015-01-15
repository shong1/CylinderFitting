# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 11:31:45 2015

@author: root
"""

import numpy as np
import math

def ransac_cylinder( pts, model, nHypoData, nTestData, max_iter, moodel_thres ):
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
	best_C = None
	best_W = None
	best_rSqr = None
	
	best_hypoErr = np.inf
	best_inlier_pts = None

	while iteration < max_iter:
		hypo_pts_idx = np.random.random_integers( 0, ptsDim[0] - 1, nHypoData )
		hypo_pts = pts[ hypo_pts_idx, : ]
		
		test_pts_idx = np.random.random_integers( 0, ptsDim[0] - 1, nTestData )
		test_pts = pts[ test_pts_idx, : ]		
		
		rSqr, C, W, minFitErr = model.fit( hypo_pts )
		tModelErr = model.get_model_err( test_pts, W, C, rSqr )
		if best_hypoErr > tModelErr:
			best_hypoErr = tModelErr
			best_C = C
			best_W = W
			best_rSqr = rSqr
		
		if best_hypoErr < model_thres:
			return best_C, best_W, best_rSqr

	return best_C, best_W, best_rSqr, best_hypoErr

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
		
		phiRange = np.arange( 0, math.pi / 2, 0.1 ) 
		thetaRange = np.arange( 0, math.pi * 2, 0.1 )
		for phi in phiRange:
			cPhi = np.cos( phi )
			sPhi = np.sin( phi )
			for theta in xrange( 0, 2 * math.pi, 0.1 ):
				cTheta = np.cos( theta )
				sTheta = np.sin( theta )
				tW = np.array( [ cTheta * sPhi, sTheta * sPhi, cPhi ] )
				tErr, tRSqr, tC = self.errorFunc( data, tW )
				
				if tErr < minFitErr:
					minFitErr = tErr
					W = tW
					C = tC
					rSqr = tRSqr
					
		C = np.add( C, meanData )
		return rSqr, C, W, minFitErr
		
	def errorFunc( self, data, W ):
		dataDim = data.shape
		
		S = np.array( [ [ 0, -W[2], W[1] ], [ W[2], 0, -W[0] ], [ -W[1], W[0], 0 ] ] )		
		P = np.subtract( np.identity( 3 ), np.outer( W, W ) )
		A = np.zeros( ( 3, 3 ) )
		B = np.zeros( ( 1, 3 ) )
		mSqrLength = 0
		sqrLength = np.zeros( ( dataDim[0], 1 ) )
		Y = np.zeros( ( dataDim[0], 3 ) )
		
		for i in xrange( dataDim[0] ):
			Yi = np.inner( P, data[ i, : ] )
			sqrLength[ i ] = np.inner( Yi, Yi )
			A += np.add( np.outer( Yi, Yi ) )
			B += sqrLength[ i ] * Yi
			mSqrLength += sqrLength[ i ]
			Y[ i, : ] = Yi
		
		A = np.divide( A, dataDim[ 0 ] )
		B = np.divide( B, dataDim[ 0 ] )
		mSqrLength = mSqrLength / dataDim[ 0 ]
		A_hat = np.inner( -S, np.inner( A, S ) )
		PC = np.divide( np.inner( A_hat, B ), np.trace( np.inner( A_hat, A ) ) )
		err = 0
		rSqr = 0
		
		for i in xrange( dataDim[0] ):
			term = sqrLength[ i ] - mSqrLength - 2 * np.dot( Y[ i ], PC )
			err += term
			diff = np.subtract( PC, Y[ i ] )
			rSqr = np.dot( diff, diff )
		
		err = err / dataDim[ 0 ]
		rSqr = rSqr / dataDim[ 0 ]		
		return err, rSqr, PC
		
	def get_model_err( self, testData, W, C, rSqr ):
		dataDim = testData.shape		
		meanData = np.mean( testData, axis = 0 )
		testData = np.subtract( testData, meanData )	
		P = np.subtract( np.identity( 3 ), np.outer( W, W ) )
		Y = np.zeros( ( dataDim[0], 3 ) )
		mSqrLength = 0
		sqrLength = np.zeros( ( dataDim[0] , 3 ) )
		
		for i in xrange( dataDim[ 0 ] ):
			Y[ i, : ] = np.inner( P, testData[ i, : ] )
			sqrLength[ i ] = np.inner( Y[ i, : ], Y[ i, : ] )
			mSqrLength = sqrLength[ i ]
			
		mSqrLength /= dataDim[ 0 ]
		err = 0
		
		for i in xrange( dataDim[ 0 ] ):
			term = sqrLength[ i ] - mSqrLength - 2 * np.dot( Y[ i ], C )
			err += term
		
		err /= dataDim[ 0 ]
		return err