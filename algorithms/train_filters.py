#################################################################################
#################################################################################
#################################################################################
#######                                                                   #######
#######                 TRAIN RECONSTRUCTION FILTERS NN-FBP               #######
#######                                                                   #######
#################################################################################
#################################################################################
#################################################################################




####  PYTHON MODULES
import numpy as np
import warnings
import random
import sys
import os
import glob
import scipy.sparse as ss
import scipy.linalg as la
try:
    import scipy.linalg.fblas as fblas
    hasfblas = True
except:
    warnings.warn(
        'No fast FBLAS implementation found, training will be slower.')
    hasfblas = False
try:
    import numexpr
    hasne = True
except:
    warnings.warn('Numexpr not installed, training will be slower.')
    hasne = False


    
    
##########################################################
##########################################################
####                                                  ####
####                    CLASS NETWORK                 #### 
####                                                  ####
##########################################################
########################################################## 
  
class Network(object):
    ##  The neural network object that performs all training and reconstruction.

    def __init__(self, nHiddenNodes, trainFiles, valFiles):
        self.tD = trainFiles
        self.vD = valFiles
        self.nHid = nHiddenNodes
        f = np.load(trainFiles[0])
        self.nIn = f.shape[1] - 1
        self.jacDiff = np.zeros((self.nHid) * (self.nIn + 1) + self.nHid + 1)
        self.jac2 = np.zeros(
            ((self.nHid) * (self.nIn + 1) + self.nHid + 1, (self.nHid) * (self.nIn + 1) + self.nHid + 1))

        
    def __inittrain(self):
        '''Initialize training parameters, create actual training and validation
        sets by picking random pixels from the datasets'''
        self.l1 = 2 * np.random.rand(self.nIn + 1, self.nHid) - 1
        beta = 0.7 * self.nHid ** (1. / (self.nIn))
        l1norm = np.linalg.norm(self.l1)
        self.l1 *= beta / l1norm
        self.l2 = 2 * np.random.rand(self.nHid + 1) - 1
        self.l2 /= np.linalg.norm(self.l2)
        self.minl1 = self.l1.copy()
        self.minl2 = self.l2.copy()
        self.minmax = self.__getMinMax()
        self.ident = np.eye((self.nHid) * (self.nIn + 1) + self.nHid + 1)

        
    def __getMinMax(self):
        minL = np.empty(self.nIn)
        minL.fill(np.inf)
        maxL = np.empty(self.nIn)
        maxL.fill(-np.inf)
        maxIn = -np.inf
        minIn = np.inf
        for i in xrange(len(self.tD)):
            data = np.load(self.tD[i])
            if data == None:
                continue
            maxL = np.maximum(maxL, data[:, 0:self.nIn].max(0))
            minL = np.minimum(maxL, data[:, 0:self.nIn].min(0))
            maxIn = np.max([maxIn, data[:, self.nIn].max()])
            minIn = np.min([minIn, data[:, self.nIn].min()])
        return (minL, maxL, minIn, maxIn)

        
    def __sigmoid(self, x):
        '''Sigmoid function'''
        if hasne:
            return numexpr.evaluate("1./(1.+exp(-x))")
        else:
            return 1. / (1. + np.exp(-x))

            
    def __createFilters(self):
        '''After training, creates the actual filters and offsets by undoing the scaling.'''
        self.minL = self.minmax[0]
        self.maxL = self.minmax[1]
        self.minIn = self.minmax[2]
        self.maxIn = self.minmax[3]
        mindivmax = self.minL / (self.maxL - self.minL)
        mindivmax[np.isnan(mindivmax)] = 0
        mindivmax[np.isinf(mindivmax)] = 0
        divmaxmin = 1. / (self.maxL - self.minL)
        divmaxmin[np.isnan(divmaxmin)] = 0
        divmaxmin[np.isinf(divmaxmin)] = 0
        self.filterWeights = np.empty((self.nHid, self.nIn))
        self.offsets = np.empty(self.nHid)
        for i in xrange(self.nHid):
            self.filterWeights[i] = 2 * self.l1[
                0:self.l1.shape[0] - 1, i] * divmaxmin
            self.offsets[i] = 2 * np.dot(
                self.l1[0:self.l1.shape[0] - 1, i], mindivmax) + np.sum(self.l1[:, i])

            
    def __processDataBlock(self, data):
        ''' Returns output values (``vals``), 'correct' output values (``valOut``) and
        hidden node output values (``hiddenOut``) from a block of data.'''
        tileM = np.tile(self.minmax[0], (data.shape[0], 1))
        maxmin = np.tile(self.minmax[1] - self.minmax[0], (data.shape[0], 1))
        data[:, 0:self.nIn] = 2 * (data[:, 0:self.nIn] - tileM) / maxmin - 1
        data[:, self.nIn] = 0.25 + (data[:, self.nIn] - self.minmax[2]) / (
            2 * (self.minmax[3] - self.minmax[2]))

        valOut = data[:, -1].copy()
        data[:, -1] = -np.ones(data.shape[0])
        hiddenOut = np.empty((data.shape[0], self.l1.shape[1] + 1))
        hiddenOut[:, 0:self.l1.shape[1]] = self.__sigmoid(
            np.dot(data, self.l1))
        hiddenOut[:, -1] = -1
        rawVals = np.dot(hiddenOut, self.l2)
        vals = self.__sigmoid(rawVals)
        return vals, valOut, hiddenOut

        
    def __getTSE(self, dat):
        '''Returns the total squared error of a data block'''
        tse = 0.
        for i in xrange(len(dat)):
            data = np.load(dat[i])
            vals, valOut, hiddenOut = self.__processDataBlock(data)
            if hasne:
                tse += numexpr.evaluate('sum((vals - valOut)**2)')
            else:
                tse += np.sum((vals - valOut) ** 2)
        return tse

        
    def __setJac2(self):
        '''Calculates :math:`J^T J` and :math:`J^T e` for the training data.
        Used for Levenberg-Marquardt method.'''
        self.jac2.fill(0)
        self.jacDiff.fill(0)
        for i in xrange(len(self.tD)):
            data = np.load(self.tD[i])
            vals, valOut, hiddenOut = self.__processDataBlock(data)
            if hasne:
                diffs = numexpr.evaluate('valOut - vals')
            else:
                diffs = valOut - vals
            jac = np.empty(
                (data.shape[0], (self.nHid) * (self.nIn + 1) + self.nHid + 1))
            if hasne:
                d0 = numexpr.evaluate('-vals * (1 - vals)')
            else:
                d0 = -vals * (1 - vals)
            ot = (np.outer(d0, self.l2))
            if hasne:
                dj = numexpr.evaluate('hiddenOut * (1 - hiddenOut) * ot')
            else:
                dj = hiddenOut * (1 - hiddenOut) * ot
            I = np.tile(
                np.arange(data.shape[0]), (self.nHid + 1, 1)).flatten('F')
            J = np.arange(data.shape[0] * (self.nHid + 1))
            Q = ss.csc_matrix((dj.flatten(), np.vstack((J, I))), (
                data.shape[0] * (self.nHid + 1), data.shape[0]))
            jac[:, 0:self.nHid + 1] = ss.spdiags(
                d0, 0, data.shape[0], data.shape[0]).dot(hiddenOut)
            Q2 = np.reshape(
                Q.dot(data), (data.shape[0], (self.nIn + 1) * (self.nHid + 1)))
            jac[:, self.nHid + 1:jac.shape[1]] = Q2[
                :, 0:Q2.shape[1] - (self.nIn + 1)]
            if hasfblas:
                self.jac2 += fblas.dgemm(1.0, a=jac.T, b=jac.T, trans_b=True)
                self.jacDiff += fblas.dgemv(1.0, a=jac.T, x=diffs)
            else:
                self.jac2 += np.dot(jac.T, jac)
                self.jacDiff += np.dot(jac.T, diffs)

                
    def train(self):
        '''Train the network using the Levenberg-Marquardt method.'''
        self.__inittrain()
        mu = 100000.
        muUpdate = 10
        prevValError = np.Inf
        bestCounter = 0
        tse = self.__getTSE(self.tD)
        for i in xrange(1000000):
            self.__setJac2()
            dw = - \
                la.cho_solve(
                    la.cho_factor(self.jac2 + mu * self.ident), self.jacDiff)
            done = -1
            while done <= 0:
                self.l2 += dw[0:self.nHid + 1]
                for k in xrange(self.nHid):
                    start = self.nHid + 1 + k * (self.nIn + 1)
                    self.l1[:, k] += dw[start:start + self.nIn + 1]
                newtse = self.__getTSE(self.tD)
                if newtse < tse:
                    if done == -1:
                        mu /= muUpdate
                    if mu <= 1e-100:
                        mu = 1e-99
                    done = 1
                else:
                    done = 0
                    mu *= muUpdate
                    if mu >= 1e20:
                        done = 2
                        break
                    self.l2 -= dw[0:self.nHid + 1]
                    for k in xrange(self.nHid):
                        start = self.nHid + 1 + k * (self.nIn + 1)
                        self.l1[:, k] -= dw[start:start + self.nIn + 1]
                    dw = - \
                        la.cho_solve(
                            la.cho_factor(self.jac2 + mu * self.ident), self.jacDiff)
            gradSize = np.linalg.norm(self.jacDiff)
            if done == 2:
                break
            curValErr = self.__getTSE(self.vD)
            if curValErr > prevValError:
                bestCounter += 1
            else:
                prevValError = curValErr
                self.minl1 = self.l1.copy()
                self.minl2 = self.l2.copy()
                if (newtse / tse < 0.999):
                    bestCounter = 0
                else:
                    bestCounter += 1
            if bestCounter == 50:
                break
            if(gradSize < 1e-8):
                break
            tse = newtse
            print 'Validation set error:', prevValError, 'Training set error:', newtse
        self.l1 = self.minl1
        self.l2 = self.minl2
        self.valErr = prevValError
        self.__createFilters()

        
    def saveToDisk(self, fn):
        ##  Save a trained network to disk, so that it can be used later
        ##  without retraining.
        ##  param fn: Filename to save it to.
        ##  type fn: :class:`string`
        np.savez(fn, filters=self.filterWeights, offsets=self.offsets,
                 weights=self.l2, minIn=self.minIn, maxIn=self.maxIn)




##########################################################
##########################################################
####                                                  ####
####                       MAIN                       ####
####                                                  ####
##########################################################
##########################################################
       
def main():
    ##  Read config file
    if len( sys.argv ) < 2:
        sys.exit( '\nERROR: Missing input config file .cfg!\n' )
    else:
        cfg_file = open( sys.argv[1] , 'r' )
        exec cfg_file
        cfg_file.close()
        
    
    ##  Get training files
    flist = sorted( glob.glob( train_path + '*.npy') ) 
    random.shuffle( flist )
    
    
    ##  Initialize class network
    perc_val = perc_val/100.0
    nval = int( len( flist ) * perc_val )
    n = Network( num_hidden_nodes , flist[nval:len(fls)], flist[0:nval])

    
    ##  Launch training
    n.train()
    

    ##  Save trained filters
    n.saveToDisk( train_path + file_trained_filters )
    
    
    
    
##########################################################
##########################################################
####                                                  ####
####                  CALL TO MAIN                    ####
####                                                  ####
##########################################################
##########################################################  

if __name__ == '__main__':
    main()    
