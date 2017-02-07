#################################################################################
#################################################################################
#################################################################################
#######                                                                   #######
#######                           NN-FBP RECONSTRUCTION                   #######
#######                                                                   #######
#################################################################################
#################################################################################
#################################################################################




####  PYTHON MODULES
from __future__ import division , print_function
import time
import datetime
import argparse
import sys
import os
import glob
import numpy as np
import multiprocessing as mproc    




####  MY PYTHON MODULES
import my_image_io as io



####  MY PROJECTOR CLASS
import class_projectors_grid as cpj 




####  NNFBP UTILITIES
import utils




####  MY FORMAT VARIABLES & CONSTANTS
myfloat = np.float32
myint = np.int 




##########################################################
##########################################################
####                                                  ####
####                        SIGMOID                   ####
####                                                  ####
##########################################################
##########################################################

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

    
    

##########################################################
##########################################################
####                                                  ####
####             MULTI-THREAD RECONSTRUCTION          ####
####               WITH CUSTOMIZED FILTERS            ####
####                                                  ####
##########################################################
##########################################################

def reconstr_nnfbp( target_path , output_path , filein , angles , ctr , 
                    weights , offsets , minIn , maxIn , NHidden , filters ):  
    ##  Read low-quality sinogram
    sino = io.readImage( target_path + filein ).astype( myfloat )
    nang , npix = sino.shape
    
        
    ##  Allocate array for reconstruction
    reco = np.zeros( ( npix , npix ) , dtype=myfloat ) 
        
    
    ##  Do the required multiple reconstructions
    for i in xrange( NHidden ):
        filt = filters[i,0:filters.shape[1]]
        tp = cpj.projectors( sino.shape[1] , angles , ctr=ctr , filt=filt )
        hidRec = tp.fbp( sino )
        reco += weights[i] * sigmoid( hidRec - offsets[i] )
        
        
    ##  Apply last sigmoid
    reco = sigmoid( reco - weights[-1] )

        
    ##  Adjust image range
    reco = 2 * ( reco - 0.25 ) * ( maxIn - minIn ) + minIn


    ##  Save reconstruction
    ext     = filein[len(filein)-4:]
    fileout = output_path + filein[:len(filein)-4] + '_nnreco' + ext
    io.writeImage( fileout , reco )
    print( '\nSaving NN-FBP reconstruction in:\n' , fileout )
    
    return



    
##########################################################
##########################################################
####                                                  ####
####                       MAIN                       ####
####                                                  ####
##########################################################
##########################################################

def main():
    ##  Initial print
    print('\n')
    print('########################################################')
    print('###                 NN-FBP RECONSTRUCTION            ###')
    print('########################################################')
    print('\n')

    
    
    ##  Read config file
    if len( sys.argv ) < 2:
        sys.exit( '\nERROR: Missing input config file .cfg!\n' )
    else:
        cfg_file = open( sys.argv[1] , 'r' )
        exec( cfg_file )
        cfg_file.close()
        
    
    ##  Get trained filters
    train_path = utils.analyze_path( train_path , mode='check' )
    ft = np.load( train_path + file_trained_filters )
    print( '\nRead file of trained filters:\n' , train_path + file_trained_filters )

    
    ##  Get low-quality sinograms
    cwd = os.getcwd()
    target_path = utils.analyze_path( target_path , mode='check' )
    
    os.chdir( target_path )    
    file_list = []
    file_list.append( sorted( glob.glob( '*' + input_files_lq + '*' ) ) )
    os.chdir( cwd )
    
    nfiles = len( file_list )
    if nfiles == 0:
        sys.exit( '\nERROR: No file *' + input_files_lq + '* found!\n' )
        
    
    ##  Check/create output path
    output_path = utils.analyze_path( output_path , mode='create' )
        

    ##  Read one sinogram
    sino = io.readImage( target_path + file_list[0][0] )
    nang , npix = sino.shape
    
    
    ##  Create angles
    angles = utils.create_projection_angles( nang=nang )

        
    ##  Create filters     
    fW      = ft['filters']
    weights = ft['weights']
    offsets = ft['offsets']
    minIn   = ft['minIn']
    maxIn   = ft['maxIn']           

    fsize = 2 * ( 2**int( np.ceil( np.log2( npix ) ) ) )
    basis = utils.generateFilterBasis( fsize , 2 ).astype( myfloat )
    
    NHidden = fW.shape[0]
    filters = np.zeros( ( NHidden , fsize ))
    for i in xrange( NHidden ):
        for j in xrange( basis.shape[0] ):
            filters[i] += basis[j] * fW[i,j]


    ##  Reconstruct low-quality sinograms
    print( '\nNN-FBP reconstruction ....' )
    ncores_avail = mproc.cpu_count
    if ncores > ncores_avail:
        ncores =  ncores_avail     
    
    pool = mproc.Pool( processes=ncores )
    for i in range( nfiles ):
        pool.apply_async( reconstr_nnfbp , args=( target_path , output_path , file_list[0][i] , angles , ctr_lq , 
                                                  weights , offsets , minIn , maxIn , NHidden , filters  ) )
    pool.close()
    pool.join() 
    
    #for i in range( nfiles ):
    #    reconstr_nnfbp( target_path , output_path , file_list[0][i] , angles , ctr_lq , 
    #                    weights , offsets , minIn , maxIn , NHidden , filters )

    print( '\n' )    

    
    
        
##########################################################
##########################################################
####                                                  ####
####               CALL TO MAIN                       ####
####                                                  ####
##########################################################
##########################################################  

if __name__ == '__main__':
    main()
