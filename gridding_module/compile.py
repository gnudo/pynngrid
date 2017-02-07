import os
import shutil


if os.path.exists('build'):
    shutil.rmtree('build')

if os.path.isfile( 'gridrec.c' ) is True:
    os.remove( 'gridrec.c' )

if os.path.isfile( 'gridrec.so' ) is True:
    os.remove( 'gridrec.so' )   

os.system('python create_module.py build_ext --inplace')
