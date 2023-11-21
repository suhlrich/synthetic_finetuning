# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:01:58 2023

@author: suhlrich
"""

import json
import os
import torch
import glob
import numpy as np

import utilsDataman
import reconstruction
from constants import AUGMENTED_VERTICES_INDEX_DICT, OPENPOSE_VERTICES_NAME

# from slahmr
from slahmr.slahmr.geometry.plane import parse_floor_plane, get_plane_transform
from slahmr.slahmr.body_model import SMPL_JOINTS, KEYPT_VERTS, smpl_to_openpose, run_smpl
from slahmr.slahmr.util.loaders import load_smpl_body_model


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

def to_torch(obj):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float()
    if isinstance(obj, dict):
        return {k: to_torch(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_torch(x) for x in obj]
    return obj

def get_results_paths(res_dir):
    """
    get the iterations of all saved results in res_dir
    :param res_dir (str) result dir
    returns a dict of iter to result path
    """
    res_files = sorted(glob.glob(f"{res_dir}/*_results.npz"))
    print(f"found {len(res_files)} results in {res_dir}")
    

    path_dict = {}
    for res_file in res_files:
        it, name, _ = os.path.basename(res_file).split("_")[-3:]
        assert name in ["world", "prior"]
        if it not in path_dict:
            path_dict[it] = {}
        path_dict[it][name] = res_file
    return path_dict

def load_result(res_path_dict):
    """
    load all saved results for a given iteration
    :param res_path_dict (dict) paths to relevant results
    returns dict of results
    """
    res_dict = {}
    for name, path in res_path_dict.items():
        res = np.load(path)
        res_dict[name] = to_torch({k: res[k] for k in res.files})
    return res_dict

# Add openpose joints
# from SLAHMR optim base_scene

def pred_smpl(body_model,trans, root_orient, body_pose, betas):
    """
    Forward pass of the SMPL model and populates pred_data accordingly with
    joints3d, verts3d, points3d.

    trans : B x T x 3
    root_orient : B x T x 3
    body_pose : B x T x J*3
    betas : B x D
    """
    smpl2op_map = smpl_to_openpose(
        'smplx',
        use_hands=False,
        use_face=False,
        use_face_contour=False,
        openpose_format="coco25",
    ) 
    
    smpl_out = run_smpl(body_model, trans, root_orient, body_pose, betas)
    joints3d, points3d = smpl_out["joints"], smpl_out["vertices"]

    # select desired joints and vertices
    joints3d_body = joints3d[:, :, : len(SMPL_JOINTS), :]
    extra_vertices = joints3d[:, :, len(SMPL_JOINTS):, :]
    joints3d_op = joints3d[:, :, smpl2op_map, :]
    # hacky way to get hip joints that align with ViTPose keypoints
    # this could be moved elsewhere in the future (and done properly)
    joints3d_op[:, :, [9, 12]] = (
        joints3d_op[:, :, [9, 12]]
        + 0.25 * (joints3d_op[:, :, [9, 12]] - joints3d_op[:, :, [12, 9]])
        + 0.5
        * (
            joints3d_op[:, :, [8]]
            - 0.5 * (joints3d_op[:, :, [9, 12]] + joints3d_op[:, :, [12, 9]])
        )
    )
    verts3d = points3d[:, :, KEYPT_VERTS, :]

    return {
        "points3d": points3d,  # all vertices
        "verts3d": verts3d,  # keypoint vertices
        "joints3d": joints3d_body,  # smpl joints
        "extra_vertices": extra_vertices, # extra vertices that we defined
        "joints3d_op": joints3d_op,  # OP joints
        "faces": smpl_out["faces"],  # index array of faces
    }

def get_vertices(vertex_idx,vertices):
    return np.moveaxis(
        np.array(
        [vertices[:,vertex,:] for vertex in vertex_idx.values()]
        ),(0,1),(1,0)
    )


# # # # Main
runYoni = False
runSLAHMR = True

markerNames = list(AUGMENTED_VERTICES_INDEX_DICT.keys())




if runYoni:
    # Yoni's input data from infinity
    data = loadJson(inputJsonPath)
    
    gender = data['info']['avatar_presenting_gender']
    betas = torch.tensor(data['info']['avatar_betas']).reshape(1,-1)
    
    # # delete rows if no annotation
    data['annotations'] = [entry for entry in data['annotations'] if 'quaternions' in entry]
    
    augmented_vertices= np.zeros((len(data['annotations']),len(markerNames)*3))
    for i,annotation in enumerate(data['annotations']):
        poses = reconstruction.get_poses(annotation)
        smplx_model = reconstruction.get_smplx_model(MODEL_FOLDER, gender, betas, poses)
        vertices, joints = reconstruction.get_vertices_and_joints(smplx_model, betas)
        augmented_vertices[i,:] = reconstruction.get_augmented_vertices(vertices).reshape(1,-1)
        
    outputFilePath = os.path.join(dataDir,'testMotion.trc')
    
    # write trc
    write_trc(augmented_vertices,outputFilePath,markerNames,frameRate=15,rotationAngles={'Y':90})
    

if runSLAHMR: 
    # From SLAHMR output
    resultsPath = ('C:/SharedGdrive/sparseIK/monocularModelTesting/SLAHMR/walk-all-shot-0-0-180/' +
                  'motion_chunks/walk_000200_world_results.npz')
    # resultsPath = ('C:/SharedGdrive/sparseIK/monocularModelTesting/SLAHMR/variedActivities2-all-shot-0-0-180/' +
    #               'motion_chunks/variedActivities2_000340_world_results.npz')
    result_base_path, _ = os.path.splitext(resultsPath)
    output_trc_path = result_base_path + '.trc'
    
    
    results=np.load(resultsPath)
    # result_dict = get_results_paths(resultDir)
    # results = load_result(result_dict)
    
    # Testing floor plane rotation
    root = torch.zeros((3,1),device='cpu')
    floor = parse_floor_plane(torch.tensor(np.squeeze(results['floor_plane'])))
    R, t = get_plane_transform(torch.tensor([0.0, 1.0, 0.0]), floor, root)
    
    # turn these into an euler sequence
    from scipy.spatial.transform import Rotation
    
    # Convert rotation matrix to Euler angles
    r_ground = Rotation.from_matrix(R.numpy().T)
    r_vis = Rotation.from_euler('x', 180, degrees=True)
    r = r_ground*r_vis
    groundR = r.as_euler('xyz', degrees=True) 
        
    # constants
    betas = torch.from_numpy(results['betas'])
    
    # see what is in results: results.files
    nSteps = results['pose_body'].shape[1]
    augmented_vertices= np.zeros((nSteps,len(markerNames)*3))

    # Get openpose points
    
    B,T,_ = results['trans'].shape
    
    body_model,gender = load_smpl_body_model(
                    path = os.path.join(MODEL_FOLDER,'smplx','SMPLX_MALE.npz'),
                    batch_size=B*T,
                    num_betas=16,
                    model_type="smplx",
                    use_vtx_selector=True,
                    device=None,
                    fit_gender = 'male',
                    npz_hack = False,
                    #extra_vertices = AUGMENTED_VERTICES_INDEX_DICT
                )
    
    preds = pred_smpl( body_model= body_model,
               trans = torch.from_numpy(results['trans']), 
               root_orient= torch.from_numpy(results['root_orient']), 
               body_pose=torch.from_numpy(results['pose_body']), 
               betas=torch.from_numpy(results['betas']))    
    
    marker_names = list(AUGMENTED_VERTICES_INDEX_DICT.keys()) + OPENPOSE_VERTICES_NAME
    nPerson = 0
    nOpenPose = preds['joints3d_op'].shape[2]
    nVertices = len(body_model.vertex_ids)
    marker_positions = np.hstack((
                                get_vertices(AUGMENTED_VERTICES_INDEX_DICT,
                                           preds['points3d'].detach()[nPerson,...].numpy()).reshape(
                                           (T,-1)),
                                preds['joints3d_op'].detach()[nPerson,...].numpy().reshape(
                                  (T,-1))
                                ))
 
    # write trc
    write_trc(marker_positions,output_trc_path,marker_names,frameRate=30,
              rotationAngles={'X':groundR[0],'Y':groundR[1],'Z':groundR[2]})




