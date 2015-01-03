# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 09:32:20 2014

@author: shong
"""

import numpy as np
import SimpleITK as sitk
import vtk
import pcl
import ransac

a = np.array( [ 1, 4, 5, 8 ], float )

binImg = sitk.ReadImage( '/home/shong/Research/FibersInConcrete/Yohan/data/StiffFibers_cropped/StiffFibers_croppedkMeanLabel2ndBlobCandidatesCCA.nrrd' );

#ptsImg= sitk.ReadImage( "/home/shong/Pictures/NonBlobRemovel.png" );
z = 10;

binMat = sitk.GetArrayFromImage( binImg );
binMatCh = sitk.GetArrayFromImage( binImg )[ :, :, z ];

ptMatSize = binMat.shape;

pts = pcl.PointCloud();

ptsMat = []

for x in range( 0, ptMatSize[0] ):
    for y in range( 0, ptMatSize[1] ):
        for z in range( 0, ptMatSize[2] ):
            if binMat.item( (x, y, z) ) == 54 :
                if ptsMat == [] :
                    ptsMat = [ [ x, y, z ] ];
                    ptsMat = np.array( ptsMat );
                ptsMat = np.append(ptsMat, [[x,y,z]], axis=0)

ptsMat = float32( ptsMat )



pts.from_array( ptsMat );

print('points:')
print( pts )