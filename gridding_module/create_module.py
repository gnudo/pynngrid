from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("gridrec",
                             sources=[ "gridrec.pyx" ,
                                       "gridrec_backproj.c" , 
                                       "fft.c" ,
                                       "pswf.c" ,
                                       "filters.c"],
                             include_dirs=[numpy.get_include()],libraries=['fftw3f','gcov'],extra_compile_args=['-O3','-march=native','-ffast-math','-fprofile-generate'],extra_link_args=['-fprofile-generate'])],
)

'''
import gridrec
gridrec.createFFTWWisdomFile(2016, "profile.wis")

import os
import sys
os.system(sys.executable + " profile.py")

os.remove('gridrec.so')

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("gridrec",
                             sources=[ "gridrec.pyx" ,
                                       "gridrec_backproj.c" , 
                                       "fft.c" ,
                                       "pswf.c" ,
                                       "filters.c"],
                             include_dirs=[numpy.get_include()],libraries=['fftw3f'],extra_compile_args=['-O3','-march=native','-ffast-math','-fprofile-use'],extra_link_args=['-fprofile-use'])],
)
'''
