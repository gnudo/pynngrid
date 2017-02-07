#################################################################################
#################################################################################
#################################################################################
#######                                                                   #######
#######                                RUN NN-FBP                         #######
#######                                                                   #######
#################################################################################
#################################################################################
#################################################################################




##  PYTHON MODULES
from __future__ import division , print_function
import sys , os




##  MAIN
if __name__ == '__main__':
    
    ##  Get config file
    if len( sys.argv ) < 2:
        sys.exit( '\nERROR: Missing input config file .cfg!\n' )
    cfg_file = sys.argv[1]


    ##  Step 1  -->  Create training dataset
    command = 'python -W ignore create_training_dataset.py ' + cfg_file
    print( '\n\nSTEP 1\n' , command )
    os.system( command )
    
    
    ##  Step 2  -->  Train reconstruction filters
    command = 'python -W ignore train_filters.py ' + cfg_file
    print( '\n\nSTEP 2\n' , command )
    os.system( command )    
           
    
    ##  Step 3  -->  Reconstruction with trained filters
    command = 'python -W ignore reconstruct.py ' + cfg_file
    print( '\n\nSTEP 3\n' , command )
    os.system( command )     