# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 09:32:20 2014

@author: shong
"""

import numpy as np
import SimpleITK as sitk
import vtk
import pcl


a = np.array( [ 1, 4, 5, 8 ], float )

ptsImg = sitk.ReadImage( '/home/shong/Research/FibersInConcrete/Yohan/data/StiffFibers_cropped/StiffFibers_cropped_N4Corrected.nrrd' );

sitk.Show( ptsImg, title = 'Check' )