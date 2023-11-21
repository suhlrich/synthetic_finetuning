import numpy as np
# import pyrender
import smplx
import torch
import os
import trimesh
from constants import AUGMENTED_VERTICES_INDEX_DICT, JOINT_NAMES, K1, K2

# from slahmr
from slahmr.slahmr.body_model import BodyModel




def get_axis_angle_from_ann(ann, start_index, end_index):
    quaternions = [
        ann["quaternions"][joint_name]
        for joint_name in JOINT_NAMES[start_index:end_index]
    ]
    axis_angle = []
    for quaternion in quaternions:
        norm = np.sqrt(quaternion[1] ** 2 + quaternion[2] ** 2 + quaternion[3] ** 2)
        angle = 2 * np.arctan2(
            norm,
            quaternion[0],
        )
        if angle == 0:
            axis_angle.append([0, 0, 0])
        else:
            axis_angle.append(
                [
                    angle * quaternion[1] / norm,
                    angle * quaternion[2] / norm,
                    angle * quaternion[3] / norm,
                ]
            )

    return torch.tensor(axis_angle, dtype=torch.float32).reshape([1, -1])


def get_body_pose(ann):
    return get_axis_angle_from_ann(ann, 1, 22)


def get_left_hand_pose(ann):
    return get_axis_angle_from_ann(ann, 25, 40)


def get_right_hand_pose(ann):
    return get_axis_angle_from_ann(ann, 40, 55)


def get_left_eye_pose(ann):
    return get_axis_angle_from_ann(ann, 23, 24)


def get_right_eye_pose(ann):
    return get_axis_angle_from_ann(ann, 24, 25)


def get_jaw_pose(ann):
    return get_axis_angle_from_ann(ann, 22, 23)


def get_poses(ann):
    return {
        "body_pose": get_body_pose(ann),
        "left_hand_pose": get_left_hand_pose(ann),
        "right_hand_pose": get_right_hand_pose(ann),
        "leye_pose": get_left_eye_pose(ann),
        "reye_pose": get_right_eye_pose(ann),
        "jaw_pose": get_jaw_pose(ann),
    }


def get_smplx_model(
    model_folder,
    gender,
    betas,
    poses,
    model_type='smplx'
):
    # Create a dictionary to store the keyword arguments
    kwargs = {
        'model_type': model_type,
        'gender': gender,
        'use_face_contour': False,
        'num_betas': max(betas.size()),
        'num_expression_coeffs': 10,
        'ext': 'npz',
        'betas': betas,
        'use_pca': False,
        'flat_hand_mean': True,
    }

    # Add pose-related arguments only if they exist in the poses dictionary
    pose_keys = [
        'body_pose',
        'left_hand_pose',
        'right_hand_pose',
        'jaw_pose',
        'leye_pose',
        'reye_pose'
    ]

    for key in pose_keys:
        if key in poses:
            kwargs[key] = poses[key]

    # Create and return the smplx model with the constructed kwargs
    return smplx.create(model_folder,**kwargs)



def get_vertices_and_joints(model, betas, kwargs = None):
    output = model(betas=betas, expression=None, return_verts=True,**kwargs)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    return vertices, joints


def get_augmented_vertices(vertices):
    return np.array(
        [vertices[vertex] for vertex in AUGMENTED_VERTICES_INDEX_DICT.values()]
    )


# # From SLAHMR
# def load_smpl_body_model(
#     path,
#     batch_size,
#     num_betas=16,
#     model_type="smplh",
#     use_vtx_selector=True,
#     device=None,
# ):
#     """
#     Load SMPL model
#     """
#     if device is None:
#         device = torch.device("cpu")
#     fit_gender = path.split("/")[-2]
#     return (
#         BodyModel(
#             bm_path=path,
#             num_betas=num_betas,
#             batch_size=batch_size,
#             use_vtx_selector=use_vtx_selector,
#             model_type=model_type,
#         ).to(device),
#         fit_gender,
#     )


# def show_mesh(
#     scene,
#     viewer,
#     vertices,
#     augmented_vertices,
#     model,
#     joints,
#     plot_augmented_vertices=True,
#     plot_joints=False,
#     nodes=[],
# ):
#     vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
#     tri_mesh = trimesh.Trimesh(vertices, model.faces, vertex_colors=vertex_colors)

#     mesh = pyrender.Mesh.from_trimesh(tri_mesh)
#     viewer.render_lock.acquire()
#     for node in nodes:
#         scene.remove_node(node)

#     nodes = [scene.add(mesh, "body")]

#     if plot_joints:
#         sm = trimesh.creation.uv_sphere(radius=0.005)
#         sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
#         tfs = np.tile(np.eye(4), (len(joints), 1, 1))
#         tfs[:, :3, 3] = joints
#         joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
#         nodes += [scene.add(joints_pcl, name="joints")]

#     if plot_augmented_vertices:
#         sm = trimesh.creation.uv_sphere(radius=0.01)
#         sm.visual.vertex_colors = [0.1, 0.1, 0.9, 1.0]
#         tfs = np.tile(np.eye(4), (len(augmented_vertices), 1, 1))
#         tfs[:, :3, 3] = augmented_vertices
#         joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
#         nodes += [scene.add(joints_pcl, name="vertices")]

#     viewer.render_lock.release()

#     return nodes
