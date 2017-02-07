##########################################################
##########################################################
####                                                  ####
####        CLASS TO HANDLE THE TOMOGRAPHIC           ####
####           FORWARD AND BACKPROJECTOR              ####
####                                                  ####
##########################################################
########################################################## 




####  PYTHON MODULES
from __future__ import division , print_function
import sys
import numpy as np




####  MY GRIDREC MODULE
cpath = '../gridrec_module/'
sys.path.append( cpath );
sys.path.append( cpath + 'pymodule_gridrec_v4/' )
import gridrec_lut as glut
import gridrec_v4 as grid
import utils




####  MY FORMAT VARIABLES
myfloat = np.float32
myint = np.int




####  LIST OF FILTERS
filter_list = np.array( [ ['none',''] , ['ramp',''] , ['shepp-logan','shlo'] , 
                          ['hanning','hann'] , ['hamming','hamm']    ,
                          ['lanczos','lanc'] , ['parzen','parz' ]      ] )




####  CLASS PROJECTORS
class projectors:

    ##  Init class projectors
    def __init__( self , npix , angles , ctr=0.0 , kernel='kb' , oversampl=1.25 , interp='lin' ,
                  radon_degree=0 , W=7.0 , errs=1e-3 , filt='parzen' , args=None ):  
                 
        ##  Compute regridding look-up-table and deapodizer
        W , lut , deapod = glut.configure_regridding( npix , kernel , oversampl , interp , W , errs )
    
        
        ##  Assign parameters
        if interp == 'nn':
            interp1 = 0
        else:
            interp1 = 1

        
        ##  Filter flag
        self.filt_ext = False
        
        if filt not in filter_list:
            self.filt_ext = True
            self.filt     = filt
            
        else:
            filt = np.argwhere( filter_list == filt )[0][0]


        ##  Setting for forward nad backprojection
        param_nofilt = np.array( [ ctr , 0 , 0 , oversampl , interp1 , 
                                   len( lut ) - 5 , W , radon_degree ] )

        
        ##  Setting for filtered backprojection
        if self.filt_ext is False:
            param_filt = np.array( [ ctr , filt , 0 , oversampl , interp1 ,
                                     len( lut ) - 5 , W , radon_degree ] )
        else:
            param_filt = param_nofilt[:]

       
        ##  Convert all input arrays to float 32
        self.lut          = lut.astype( myfloat )
        self.deapod       = deapod.astype( myfloat )
        self.angles       = angles.astype( myfloat )
        self.param_nofilt = param_nofilt.astype( myfloat )
        self.param_filt   = param_filt.astype( myfloat )
    

    
    ##  Forward projector
    def A( self , x ):
        return grid.forwproj( x.astype( myfloat ) , self.angles , self.param_nofilt ,
                              self.lut , self.deapod )


    
    ##  Backprojector
    def At( self , x ):
        return grid.backproj( x.astype( myfloat ) , self.angles , self.param_nofilt , 
                              self.lut , self.deapod )


    
    ##  Filtered backprojection (with ramp filter as default)
    def fbp( self , x ):
        if self.filt_ext is True:
            x[:] = utils.filter_proj_custom( x , self.filt )            
        return grid.backproj( x , self.angles , self.param_filt ,
                              self.lut , self.deapod )  
