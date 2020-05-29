# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:37:37 2020

@author: askou
"""

# =============================================================================
# try:
#     from IPython import get_ipython
#     get_ipython().magic('clear')
#     get_ipython().magic('reset -f')
# except:
#     pass
# =============================================================================

import scipy.io
import numpy as np
# import scipy.stats as stt
import matplotlib.pyplot as plt # plots
import os # operating system interface
# from pymou.mou_model import MOU

plt.close('all')

## Create a local folder to store results
resDir = 'modeParameter/'
if not os.path.exists(resDir):
    print('created directory:',resDir)
    os.makedirs(resDir)

## Read in data, structure it, define constants
wholeStruct = scipy.io.loadmat('DATA_TASK_3DMOV_HP_CSF_WD.mat')
dataStruct = wholeStruct['TASKEC'][0][0]
roiLabels = wholeStruct['ROIlbls'][0]
rest = dataStruct['Rest']
nBack = dataStruct['nBack']
flanker = dataStruct['Flanker']
mRotation = dataStruct['mRotation']
oddManOut = dataStruct['OddManOut']

nSubjects = rest.shape[2]
nRuns = len(wholeStruct['TASKEC'][0][0])
nROIs = rest.shape[1]
nTsSamples = rest.shape[0]
timeResolution = 2.0

filteredTsEmp = np.zeros([nSubjects,nRuns,nROIs,nTsSamples])
run = list(dataStruct.dtype.fields.keys())
# =============================================================================
# test = np.zeros([len(run)])
# =============================================================================
for i in range(len(run)):
    filteredTsEmp[:,i,:,:] = np.transpose(dataStruct[run[i]],(2,1,0))
# =============================================================================
#     for j in range(nTsSamples):
#         for k in range(nROIs):
#             for l in range(nSubjects):
#                 test[i] = dataStruct[run[i]][j][k][l] == filteredTsEmp[l][i][k][j]
#                 if test[i] == False: break
# print('The transpose function works as I supposed it would: ', str(test))
# =============================================================================


## Calculate functional connectivity (BOLD covariances) [Q0 und Q1]
timeShift = np.arange(4,dtype=float)
nShifts = len(timeShift)

fcEmp = np.zeros([nSubjects,nRuns,nShifts,nROIs, nROIs])
for iSubject in range(nSubjects):
    for iRun in range (nRuns):
        # center the time series
        filteredTsEmp[iSubject,iRun,:,:] -=  \
                np.outer( filteredTsEmp[iSubject,iRun,:,:].mean(1), np.ones([nTsSamples]) )
        for iShift in range(nShifts):
            fcEmp[iSubject, iRun, iShift, :, :] = \
                np.tensordot(filteredTsEmp[iSubject,iRun,:,
                                           0:nTsSamples-nShifts+1],
                             filteredTsEmp[iSubject,iRun,:,
                                           iShift:nTsSamples-nShifts+1+iShift],
                             axes = (1,1)) / float((nTsSamples-nShifts))

# implement the calculation of the covariance by hand
fcEmpTest = np.zeros([nSubjects,nRuns,nShifts,nROIs, nROIs])
for iSubject in range(nSubjects):
    for iRun in range (nRuns):
        for iShift in range(nShifts):
            for iROI in range(nROIs):
                fcEmpTest[iSubject, iRun, iShift, :, iROI] = \
                    np.dot(filteredTsEmp[iSubject,iRun,
                                         :,0:nTsSamples-nShifts+1],
                    filteredTsEmp[iSubject,iRun,
                                  iROI,iShift:nTsSamples-nShifts+1+iShift])/ \
                                         float((nTsSamples-nShifts))#.reshape(nTsSamples-nShifts+1+iShift,1))

# rescale_FC_factor = 0.12127795556096475
rescaleFcFactor = 0.5 / fcEmp[:,0,0,:,:].diagonal(axis1=1, axis2=2).mean()
#rescaleFcFactor = 1 / (nTsSamples - 1) # with this rescaleFactor the values are not normalized (vmax=1 is too high as the highest values are ~0.0017)
fcEmp *= rescaleFcFactor

filteredTsEmp /= np.sqrt(rescaleFcFactor)

print('most of the FC values should be between 0 and 1')
print('mean FC0 value:', fcEmp[:,:,0,:,:].mean())
print('max FC0 value:', fcEmp[:,:,0,:,:].max())
print('mean BOLD variance (diagonal of each FC0 matrix):',
      fcEmp[:,:,0,:,:].diagonal(axis1=2,axis2=3).mean())

# Show distibution of FC0 values
plt.figure()
plt.hist(fcEmp[:,:,0,:,:].flatten(), bins=np.linspace(-1,5,30))
plt.xlabel('FC0 value', fontsize=14)
plt.ylabel('matrix element count', fontsize=14)
plt.title('distribution of FC0 values')

# Show FC0 averaged over subjects, first run (rest)
plt.figure()
plt.imshow(fcEmp[:,0,0,:,:].mean(axis=0), origin = 'lower',
           cmap=plt.cm.get_cmap('Blues',256), vmax=1)
#plt.imshow(fcEmp[1,0,0,:,:], origin = 'lower', cmap='Blues', vmax = 1)
plt.colorbar()
plt.xlabel('source ROI', fontsize=14)
plt.ylabel('target ROI', fontsize=14)
plt.title('FC0 (functional connectivity with no time lag)')
plt.show()

# Show FC0 of first subject, first run (rest)
plt.figure()
plt.imshow(fcEmp[0,0,0,:,:].mean(axis=0), origin = 'lower',
           cmap=plt.cm.get_cmap('Blues',256), vmax=1)
plt.colorbar()
plt.xlabel('source ROI', fontsize=14)
plt.ylabel('target ROI', fontsize=14)
plt.title('FC0 (functional connectivity with no time lag)')
plt.show()

# Show Test FC0 averaged over subjects, first run (rest)
plt.figure()
plt.imshow(fcEmpTest[:,0,0,:,:].mean(axis=0), origin = 'lower',
           cmap=plt.cm.get_cmap('Blues',256), vmax=1)
#plt.imshow(fcEmp[1,0,0,:,:], origin = 'lower', cmap='Blues', vmax = 1)
plt.colorbar()
plt.xlabel('source ROI', fontsize=14)
plt.ylabel('target ROI', fontsize=14)
plt.title('Test FC0 (functional connectivity with no time lag)')
plt.show()

# Show FC1 averaged over subjects, first run (rest)
plt.figure()
plt.imshow(fcEmp[:,0,1,:,:].mean(axis=0), origin = 'lower',
           cmap='Blues', vmax = 1)
plt.colorbar()
plt.xlabel('source ROI', fontsize=14)
plt.ylabel('target ROI', fontsize=14)
plt.title('FC1 (functional connectivity with no time lag)')

plt.show()

# show autocovariance
ac = fcEmp.diagonal(axis1=3,axis2=4)
# structure of fcEmp: [nSubjects,nRuns,nShifts,nROIs, nROIs]

# =============================================================================
# for iSubject in range(nSubjects):
#     plt.figure()
#     plt.plot(range(4), np.log10(np.maximum(ac[iSubject,0,:,:],
#                                 1e-3)))#np.exp(-3.0))))
#     plt.xlabel('time lag', fontsize=14)
#     plt.ylabel('log autocovariance', fontsize=14)
#     plt.title('rest', fontsize=16)
#
#     plt.figure()
#     plt.plot(range(4), np.log(np.maximum(ac[iSubject,1,:,:],0.05))) # or rather range(n_tau)
#     plt.xlabel('time lag', fontsize=14)
#     plt.ylabel('log autocovariance', fontsize=14)
#     plt.title('nBack', fontsize=16)
# plt.show()
# =============================================================================

plt.figure()
plt.plot(range(nShifts), np.log10(np.maximum(ac[:,0,:,:].mean(axis=0),
                                (1e-3)))) # average over the subjects
plt.xlabel('time lag', fontsize=14)
plt.ylabel('log autocovariance', fontsize=14)
plt.title('rest', fontsize=16)

plt.figure()
plt.plot(range(nShifts), np.log10(np.maximum(ac[:,1,:,:].mean(axis=0),
                                (1e-3)))) # average over the subjects
plt.xlabel('time lag', fontsize=14)
plt.ylabel('log autocovariance', fontsize=14)
plt.title('nBack', fontsize=16)

plt.show()

plt.figure()
plt.plot(range(nShifts), np.log2(np.maximum(ac[:,0,:,:].mean(axis=0),
                                (2**-3)))) # average over the subjects
plt.xlabel('time lag', fontsize=14)
plt.ylabel('log autocovariance', fontsize=14)
plt.title('rest', fontsize=16)

plt.figure()
plt.plot(range(nShifts), np.log2(np.maximum(ac[:,1,:,:].mean(axis=0),
                                (2**-3)))) # average over the subjects
plt.xlabel('time lag', fontsize=14)
plt.ylabel('log autocovariance', fontsize=14)
plt.title('nBack', fontsize=16)

plt.show()

plt.figure()
plt.plot(range(nShifts), np.log(np.maximum(ac[:,0,:,:].mean(axis=0),
                                np.exp(-3.0)))) # average over the subjects
plt.xlabel('time lag', fontsize=14)
plt.ylabel('log autocovariance', fontsize=14)
plt.title('rest', fontsize=16)

plt.figure()
plt.plot(range(nShifts), np.log(np.maximum(ac[:,1,:,:].mean(axis=0),
                                np.exp(-3.1)))) # average over the subjects
plt.xlabel('time lag', fontsize=14)
plt.ylabel('log autocovariance', fontsize=14)
plt.title('nBack', fontsize=16)

plt.show()
