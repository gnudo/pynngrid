#################################################################################
#################################################################################
#################################################################################
#######                                                                   #######
#######               CREATE TRAINING DATASET FOR NN-FBP                  #######
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
####             MULTI-THREAD RECONSTRUCTION          ####
####               WITH CUSTOMIZED FILTERS            ####
####                                                  ####
##########################################################
##########################################################

def reconstr_filter_custom( sino_lq , angle_lq , ctr , filt_custom , picked ):  
    ##  Load gridding projector class
    tp = cpj.projectors( sino_lq.shape[1] , angles_lq , ctr=ctr , filt=filt_custom )
    
    ##  Reconstruction
    reco = tp.fbp( sino )
    
    ##  Pick up only selected pixels
    reco = reco[ picked ]
    
    return reco



    
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
    print('###              CREATING TRAINING DATASET           ###')
    print('########################################################')
    print('\n')

    
    
    ##  Read config file
    if len( sys.argv ) < 2:
        sys.exit( '\nERROR: Missing input config file .cfg!\n' )
    else:
        cfg_file = open( sys.argv[1] , 'r' )
        exec cfg_file
        cfg_file.close()
        
    
    ##  Get list of input files
    cwd = os.getcwd()
    
    input_path = utils.analyze_path( input_path , mode='check' )
    
    os.chdir( input_path )    
    filein = sorted( glob.glob( '*' + input_files + '*' ) )
    os.chdir( cwd )
    
    nfiles = len( filein )
    if nfiles == 0:
        sys.exit( '\nERROR: No file *' + input_files_hq + '* found!\n' )
        
    train_path = utils.analyze_path( train_path , mode='create' )    
        
    print( '\nInput data folder:\n' , input_path )
    print( '\nTrain data folder:\n' , train_path )
    print( '\nInput high-quality sinograms:\n' , nfiles )
        
    
    ##  Read one file
    sino = io.readImage( input_path + filein[0][0] )
    nang , npix = sino.shape
    print( '\nSinogram(s) with ' , nang ,' views X ' , npix, ' pixels' )
    
    
    ##  Create array of projection angles
    angles = utils.create_projection_angles( nang=nang )
    
        
    ##  Load gridding projectors
    tp = cpj.projectors( npix , angles , ctr=ctr_rot , filt=filt ) 
 
    
    ##  Compute number of views for low-quality training sinograms
    nang_new = np.int( nang / ( factor_down * 1.0 ) )
    print( '\nDownsampling factor for training sinograms: ' , factor_down )
    
    
    ##  Create customized filters
    print( '\nCreating customized filters ....' )
    filt_size   = 2 * ( 2**int( np.ceil( np.log2( npix ) ) ) )
    filt_custom = nnfbp_hf.generateFilterBasis( npix , 2 )
    nfilt       = filt_custom.shape[0]
    

    ##  Region of interest to select training data
    idx = nnfbp_hf.getIDX( npix , roi_l , roi_r , roi_b , roi_t )
    
    
    ##  Create training dataset 
    print( '\nCreating training dataset ....' , end='' ) 
    
    ncores_avail = mproc.cpu_count
    if ncores > ncores_avail:
        ncores =  ncores_avail     
    
    for i in range( nfiles ):
        ##  Read high-quality sinogram
        sino_hq = io.readImage( input_path + file_list[0][i] ).astype( myfloat )

        ##  Reconstruct high-quality sinogram with standard filter
        reco_hq = tp.fbp( sino_hq )            
        
        ##  Create output training array
        train_data = np.zeros( ( npix_train_slice , nfilt+1 ) , dtype=myfloat )
        
        ##  Randomly select training pixels
        picked = nnfbp_hf.getPickedIndices( idx , npix_train_slice )
        
        ##  Save validation data
        train_data[:,-1] = reco_hq[picked]
        
        ##  Downsample sinogram
        sino_lq , angle_lq = utils.downsample_sinogram_angles( sino_hq , nang_new )
        
        ##  Reconstruct low-quality sinograms with customized filters
        pool = mproc.Pool( processes=ncores )
        results = [ pool.apply_async( reconstr_filter_custom , 
                                      args=( sino_lq , angle_lq , ctr , filt_custom[j,:] , picked ) ) \
                                      for j in range( nfilt ) ]
        train_data[:,:nfilt] = np.array( [ res.get() for res in results ] )
        pool.close()
        pool.join()

        ##  Save training data
        filename = file_list[0][i]
        fileout  = train_path + filename[:len(filename)-4] + '_train.npy'
        np.save( fileout , train_data )
        
    print( ' done!' ) 

    print( '\nTraining data saved in:\n', train_path,'\n' )    
    



##########################################################
##########################################################
####                                                  ####
####               CALL TO MAIN                       ####
####                                                  ####
##########################################################
##########################################################  

if __name__ == '__main__':
    main()
