# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:01:58 2023

@author: suhlrich
"""

import json
import os
import torch
import numpy as np

import utilsDataman
import reconstruction

from constants import MODEL_FOLDER, AUGMENTED_VERTICES_INDEX_DICT


# # # user edited

dataDir = os.path.abspath('data')
inputJsonPath = os.path.join(dataDir,'labels.json')

# # #  Functions

# load json
def loadJson(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data

def numpy2TRC(f, data, headers, fc=50.0, t_start=0.0, units="m"):
    # data -> nFrames x nMarkers*3 array
    
    header_mapping = {}
    for count, header in enumerate(headers):
        header_mapping[count+1] = header 
        
    # Line 1.
    f.write('PathFileType  4\t(X/Y/Z) %s\n' % os.getcwd())
    
    # Line 2.
    f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\t'
                'Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
    
    num_frames=data.shape[0]
    num_markers=len(header_mapping.keys())
    
    # Line 3.
    f.write('%.1f\t%.1f\t%i\t%i\t%s\t%.1f\t%i\t%i\n' % (
            fc, fc, num_frames,
            num_markers, units, fc,
            1, num_frames))
    
    # Line 4.
    f.write("Frame#\tTime\t")
    for key in sorted(header_mapping.keys()):
        f.write("%s\t\t\t" % format(header_mapping[key]))

    # Line 5.
    f.write("\n\t\t")
    for imark in np.arange(num_markers) + 1:
        f.write('X%i\tY%s\tZ%s\t' % (imark, imark, imark))
    f.write('\n')
    
    # Line 6.
    f.write('\n')

    for frame in range(data.shape[0]):
        f.write("{}\t{:.8f}\t".format(frame+1,(frame)/fc+t_start)) # opensim frame labeling is 1 indexed

        for key in sorted(header_mapping.keys()):
            f.write("{:.5f}\t{:.5f}\t{:.5f}\t".format(data[frame,0+(key-1)*3], data[frame,1+(key-1)*3], data[frame,2+(key-1)*3]))
        f.write("\n")

# write trc
def write_trc(keypoints3D, pathOutputFile, keypointNames, 
                            frameRate=60, rotationAngles={}):

    with open(pathOutputFile,"w") as f:
        numpy2TRC(f, keypoints3D, keypointNames, fc=frameRate, 
                  units="m")
    
    # Rotate data to match OpenSim conventions; this assumes the chessboard
    # is behind the subject and the chessboard axes are parallel to those of
    # OpenSim.
    trc_file = utilsDataman.TRCFile(pathOutputFile)    
    for axis,angle in rotationAngles.items():
        trc_file.rotate(axis,angle)

    trc_file.write(pathOutputFile)   
    
    return None

# # # # Main

# Yoni's input data from infinity
data = loadJson(inputJsonPath)

# gender = data['info']['avatar_presenting_gender']
# betas = torch.tensor(data['info']['avatar_betas']).reshape(1,-1)
# markerNames = list(AUGMENTED_VERTICES_INDEX_DICT.keys())

# # delete rows if no annotation
# data['annotations'] = [entry for entry in data['annotations'] if 'quaternions' in entry]

# Load output data from SLAHMR
dataPath = 'C:\SharedGdrive\sparseIK\monocularModelTesting\SLAHMR\walk-all-shot-0-0-180'
resultsPath = os.path.join(dataPath,'motion_chunks','walk_000200_world_results.npz')
results=np.load(resultsPath)

poses = results['pose_body'] # 1 x nFrames x 63??





augmented_vertices= np.ndarray((len(data['annotations']),len(markerNames)*3))
for i,annotation in enumerate(data['annotations']):
    poses = reconstruction.get_poses(annotation)
    smplx_model = reconstruction.get_smplx_model(MODEL_FOLDER, gender, betas, poses)
    vertices, joints = reconstruction.get_vertices_and_joints(smplx_model, betas)
    augmented_vertices[i,:] = reconstruction.get_augmented_vertices(vertices).reshape(1,-1)
    
outputFilePath = os.path.join(dataDir,'testMotion.trc')

# write trc
write_trc(augmented_vertices,outputFilePath,markerNames,frameRate=15,rotationAngles={'Y':90})
    





