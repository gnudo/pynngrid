###########################################################################
###########################################################################
####                                                                   ####
####                  COMPILE ALL SUBROUTINES IN C                     ####
####                                                                   ####
###########################################################################
###########################################################################

##  Usage:
##    MODE-1:     python setup.py   --> install all subroutines in C
##    MODE-2:     python setup.py 1 --> delete all compiles files '.o', '.so' and all
##                                      folders 'debug/', 'build/' and '__pycache__' 




##  Python packages
from __future__ import division , print_function
import os
import sys
import shutil




##  Choose whether to use the script in MODE-1 or in MODE-2
if len( sys.argv ) == 1:
    compile = True
else:
    compile = False


##  Remove all compile files and related folders
curr_dir = os.getcwd()

for path , subdirs , files in os.walk( './' ):
    for i in range( len( subdirs ) ):
        folderin = subdirs[i]
        if folderin == 'build' or folderin == 'debug':
            shutil.rmtree( os.path.join( path , folderin ) )
        if folderin == '__pycache__':
            shutil.rmtree( os.path.join( path , folderin ) )
    
    for i in range( len( files ) ):
        filein = files[i]

        if filein.endswith( '.pyc' ) is True or \
           filein.endswith( '.pyo' ) is True or \
           filein.endswith( '.o' )   is True or \
           filein.endswith( '.so' )  is True or \
           filein.endswith( '.swp' )  is True or \
           filein.endswith( '.swo' )  is True or \
           filein.endswith( '.swn' )  is True or \
           filein.endswith( '~' )  is True:
            os.remove( os.path.join( path , filein ) ) 


##  Compile all subroutines in C if enabled
folder1 = 'sl3d_128x128x102_ang0256'
folder2 = 'test_data'

if compile is True:
    path = 'gridding_module/'
    os.chdir( path )
    os.system( 'python compile.py' )
    os.chdir( curr_dir )

    os.chdir( 'data/' )
    os.system( 'unzip ' + folder1 + '.zip' )
    os.system( 'rm ' + folder1 + '.zip' )
    os.system( 'unzip ' + folder2 + '.zip' )
    os.system( 'rm ' + folder2 + '.zip' )    


else:
    os.chdir( curr_dir ) 
    os.chdir( 'data/' )
    if os.path.exists( folder1 ):
        os.system( 'zip -r ' + folder1 + '.zip ' + folder1 )
        shutil.rmtree( folder1 )
    os.system( 'zip -r ' + folder2 + '.zip ' + '*.DMP' )
    if os.path.exists( 'train' ):
        os.system( 'rm -r train' ) 
    os.system( 'rm *.DMP' )     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
