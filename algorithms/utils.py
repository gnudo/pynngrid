################################################################
################################################################
####                                                        ####
####                  UTILITIES FOR NN-FBP                  ####
####                                                        ####
################################################################
################################################################




####  PYTHON MODULES
from __future__ import division , print_function
import numpy as np
import sys
import os , errno
import math , random
import shutil
import glob
import bisect

try:
    import pywt
    hasPyWt=True
except:
    print ( 'PyWavelets not installed, ring removal not available!' )
    hasPyWt=False


    
    
####  MY VARIABLE TYPE
myint   = np.int
myfloat = np.float32

    


#############################################################
#############################################################
####                                                     ####
####                  CHECK FOLDER NAME                  ####
####                                                     ####
#############################################################
#############################################################

def analyze_path( pathin , mode='check' ):
    if not os.path.exists( pathin ):
        if mode is 'check':
            sys.exit( '\nERROR: Input path ' + pathin + ' does not exist!\n' )
        elif mode is 'create':
            os.makedirs( pathin )
                
    if pathin[len(pathin)-1] != '/':
        pathin += '/'
        
    return pathin

    
    
    
#############################################################
#############################################################
####                                                     ####
####                REMOVE SINOGRAM STRIPES              ####
####                                                     ####
#############################################################
#############################################################

def removeStripesTomoPy(sinogram,level=5,sigma=2.4,wname='db20'):
    #From tomopy https://github.com/tomopy/tomopy/blob/master/tomopy/preprocess/stripe_removal.py
    # 24/03/2014
    
    if not hasPyWt:
        print( 'PyWavelets not installed, ring removal not available!' )
        return sinogram
    size = np.max(sinogram.shape)
    if level==None: level = int(np.ceil(np.log2(size)))
    dx, dy = sinogram.shape
    
    num_x = dx + dx / 8
    x_shift = int((num_x - dx) / 2.)
    sli = np.zeros((num_x, dy), dtype='float32')
    
    
    sli[x_shift:dx+x_shift, :] = sinogram
    
    # Wavelet decomposition.
    cH = []
    cV = []
    cD = []
    for m in range(level):
        sli, (cHt, cVt, cDt) = pywt.dwt2(sli, wname)
        cH.append(cHt)
        cV.append(cVt)
        cD.append(cDt)

    # FFT transform of horizontal frequency bands.
    for m in range(level):
        # FFT
        fcV = np.fft.fftshift(np.fft.fft(cV[m], axis=0))
        my, mx = fcV.shape

        # Damping of ring artifact information.
        y_hat = (np.arange(-my, my, 2, dtype='float')+1) / 2
        damp = 1 - np.exp(-np.power(y_hat, 2) / (2 * np.power(sigma, 2)))
        fcV = np.multiply(fcV, np.transpose(np.tile(damp, (mx, 1))))

        # Inverse FFT.
        cV[m] = np.real(np.fft.ifft(np.fft.ifftshift(fcV), axis=0))

    # Wavelet reconstruction.
    for m in range(level)[::-1]:
        sli = sli[0:cH[m].shape[0], 0:cH[m].shape[1]]
        sli = pywt.idwt2((sli, (cH[m], cV[m], cD[m])), wname)
    return sli[x_shift:dx+x_shift, 0:dy].astype(np.float32)

def getCPUToSinogramIndices(start,end,nCpu):
    nSino = end-start+1
    if nSino<nCpu: nCpu = nSino
    cpuStartEnd = start + np.array([np.round(i*float(nSino)/nCpu) for i in xrange(nCpu+1)],dtype=np.int)
    cpuStartEnd[-1]=end+1
    return cpuStartEnd

def getIDX(size,l=None,r=None,t=None,b=None):
    '''Create a variable ``idx`` that gives location of pixels that can be picked.'''
    ym,xm = np.ogrid[-(size-1.)/2.:(size-1.)/2.:complex(0,size),-(size-1.)/2.:(size-1.)/2.:complex(0,size)]
    bnd = (size)**2/4
    mask = xm**2+ym**2<=bnd
    if not l==None:
        mask = np.logical_and(mask,xm>=l)
    if not r==None:
        mask = np.logical_and(mask,xm<=r)
    if not t==None:
        mask = np.logical_and(mask,ym>=t)
    if not b==None:
        mask = np.logical_and(mask,ym<=b)
    x,y = np.where(mask==True)
    return zip(x,y)

def getPickedIndices(idx,nToPick):
    '''Return a list of the location of ``nToPick`` randomly selected pixels.'''
    nTimesToDo = int(math.ceil(nToPick/float(len(idx))))
    iList = []
    for i in xrange(nTimesToDo):
        iList.extend(idx)
    return zip(*random.sample(iList,nToPick))

def generateFilterBasis(size,NLinear):
    cW=0
    nW=0
    width=1
    nL = NLinear
    while cW<(size-1)/2:
        if nL>0:
            nL-=1
            cW += 1
        else:
            cW += width
            width*=2
        nW+=1
        
    basis = np.zeros((nW,size))
    x = np.linspace(0,2*np.pi,size,False)
    cW=0
    nW=0
    width=1
    nL = NLinear
    while cW<(size-1)/2:
        if nL>0:
            nL-=1
            eW = cW+1
        else:
            eW = cW+width
            width*=2
        basis[nW] += np.cos(np.outer(np.arange(cW,eW),x)).sum(0)
        cW=eW
        nW+=1
    return basis



    
#############################################################
#############################################################
####                                                     ####
####               CREATE PROJECTION ANGLES              ####
####                                                     ####
#############################################################
#############################################################

def create_equally_spaced_angles( nang , angle_start , angle_end ):
    angles = np.linspace( angle_start , angle_end , nang , endpoint=False )
    return angles  



def create_pseudo_polar_angles( nang ):
    if nang % 4 != 0:
        raise Exception('\n\tError inside createPseudoPolarAngles:'
                        +'\n\t  nang (input) is not divisible by 4 !\n')
    n = nang
    pseudo_angles = np.zeros(n,dtype=myfloat)
    pseudoAlphaArr = np.zeros(n,dtype=myfloat)
    pseudoGridIndex = np.zeros(n,dtype=int)
    index = np.arange(int(n/4)+1,dtype=int)

    pseudo_angles[0:int(n/4)+1] = np.arctan(4*index[:]/myfloat(n))
    pseudo_angles[int(n/2):int(n/4):-1] = np.pi/2-pseudo_angles[0:int(n/4)]    
    pseudo_angles[int(n/2)+1:] = np.pi-pseudo_angles[int(n/2)-1:0:-1]

    pseudo_angles *= 180.0 / np.pi
    
    return pseudo_angles



##  Note: this function gives as output angles in degrees,
##        excluding the right extreme, e.g., [ 0.0 , 180 )

def create_projection_angles( nang=None , start=0.0 , end=180.0 , 
                              pseudo=0 , wedge=False , textfile=None ):
    
    ##  Create angles
    if textfile is None:
        ##  Create equally angularly space angles in [alpha_{0},alpha_{1})
        if pseudo == 0:
            if wedge is False:
                angles = create_equally_spaced_angles( nang , start , end )

            else:
                angles1 = create_equally_spaced_angles( nang , 0.0 , start )
                angles2 = create_equally_spaced_angles( nang , end , 180.0 )
                angles = np.concatenate( ( angles1 , angles2 ) , axis=1 )

        ##  Create equally sloped angles
        else:
            angles = create_pseudo_polar_angles( nang )


    ##  Read angles from text file
    else:
        angles = np.fromfile( textfile , sep="\t" ) 

    return angles

    
    

################################################################
################################################################
####                                                        ####
####                      BINARY SEARCH                     ####
####                                                        ####
################################################################
################################################################ 
    
def binary_search( array , el ):
    ind  = bisect.bisect_left( array , el )

    if array[ind] > el:
        ind_left  = ind - 1
        ind_right = ind
    else:
        ind_left  = ind
        ind_right = ind + 1  
    
    al = array[ind_left];  ar = array[ind_right]

    if np.abs( al - el ) < np.abs( ar - el ):
        return ind_left
    else:
        return ind_right




################################################################
################################################################
####                                                        ####
####             DOWNSAMPLE SINOGRAM IN ANGLES              ####
####                                                        ####
################################################################
################################################################

def downsample_sinogram_angles( sino , angles , nproj ):
    nang = len( angles )
    angles_down = create_projection_angles( nang=nproj )
    
    ii = np.zeros( nproj , dtype=int )
    
    for i in range( nproj ):
        ii[i] = binary_search( angles , angles_down[i] )
    
    sino_down   = sino[ii,:]
    angles_down = angles[ii]

    return sino_down , angles_down
    
    
    
    
##########################################################
##########################################################
####                                                  ####
####                 PROJECTION FILTERING             ####
####                                                  ####
##########################################################
##########################################################

def filter_proj_custom( sino , filt ):
    ##  Get dimensions
    nang , npix = sino.shape
    nfreq = len( filt )
    
    
    ##  Zero-pad projections    
    sino_pad = np.concatenate( ( sino , np.zeros( ( nang , nfreq - npix ) ) ) , axis=1 ) 
            
    
    ##  Filtering in Fourier space    
    for i in range( nang ):
        sino_pad[i,:] = np.real( np.fft.ifft( np.fft.fft( sino_pad[i,:] ) * filt ) )                


    ##  Replace values in the original array
    sino[:,:] = sino_pad[:,:npix]

    return sino