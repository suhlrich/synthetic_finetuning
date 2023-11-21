import csv

import numpy as np


def load_augmented_corr():
    with open(AUGMENTED_VERTICES_FILE_PATH, "r", encoding="utf-8-sig") as data:
        augmented_vertices_index = list(csv.DictReader(data))
        augmented_vertices_index_dict = {
            vertex["Name"]: int(vertex["Index"]) for vertex in augmented_vertices_index
        }

    return augmented_vertices_index_dict


exercises = [
    "V_UP",
    "SITUP",
    "BRIDGE",
    "BURPEE",
    "PUSHUP",
    "BIRD_DOG",
    "CRUNCHES",
    "SUPERMAN",
    "LEG_RAISE",
    "DOWNWARD_DOG",
    "UPPERCUT-LEFT",
    "CLAMSHELL-LEFT",
    "UPPERCUT-RIGHT",
    "CLAMSHELL-RIGHT",
    "LUNGE-CROSSBACK",
    "BEAR_CRAWL-HOLDS",
    "DONKEY_KICK-LEFT",
    "PUSHUP-EXPLOSIVE",
    "SQUAT-BODYWEIGHT",
    "DEADLIFT-DUMBBELL",
    "DONKEY_KICK-RIGHT",
    "PUSHUP-CLOSE_GRIP",
    "ARM_RAISE-DUMBBELL",
    "BICEP_CURL-BARBELL",
    "SQUAT-BACK-BARBELL",
    "SQUAT-GOBLET+SUMO-DUMBBELL",
    "PRESS-SINGLE_ARM-DUMBBELL-LEFT",
    "BICEP_CURL-ALTERNATING-DUMBBELL",
    "PRESS-SINGLE_ARM-DUMBBELL-RIGHT",
    "PUSH_PRESS-SINGLE_ARM-DUMBBELL-LEFT",
    "PUSH_PRESS-SINGLE_ARM-DUMBBELL-RIGHT",
    "SPLIT_SQUAT-SINGLE_ARM-DUMBBELL-LEFT",
    "SPLIT_SQUAT-SINGLE_ARM-DUMBBELL-RIGHT",
    "TRICEP_KICKBACK-BENT_OVER+SINGLE_ARM-DUMBBELL-LEFT",
    "TRICEP_KICKBACK-BENT_OVER+SINGLE_ARM-DUMBBELL-RIGHT",
]


K1 = np.array([[311.11, 0.0, 112.0], [0.0, 311.11, 112.0], [0.0, 0.0, 1.0]])
K2 = np.array([[245.0, 0.0, 112.0], [0.0, 245.0, 112.0], [0.0, 0.0, 1.0]])

AUGMENTED_VERTICES_FILE_PATH = "data/vertices_keypoints_corr.csv"
AUGMENTED_VERTICES_INDEX_DICT = load_augmented_corr()
AUGMENTED_VERTICES_NAMES = list(AUGMENTED_VERTICES_INDEX_DICT.keys())
COCO_VERTICES_NAME = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

OPENPOSE_VERTICES_NAME = [
    'Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow',
    'LWrist', 'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'REye', 'LEye', 'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe',
    'RSmallToe', 'RHeel'
]

MODEL_FOLDER = "./models"


JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]

JSON_CATEGORIES = [
    {
        "id": 0,
        "augmented_keypoints": [
            "sternum",
            "rshoulder",
            "lshoulder",
            "r_lelbow",
            "l_lelbow",
            "r_melbow",
            "l_melbow",
            "r_mwrist",
            "l_mwrist",
            "r_lwrist",
            "l_lwrist",
            "C7",
            "r_ASIS",
            "l_ASIS",
            "r_PSIS",
            "l_PSIS",
            "r_knee",
            "r_mknee",
            "l_knee",
            "l_mknee",
            "r_ankle",
            "r_mankle",
            "l_ankle",
            "l_mankle",
            "r_5meta",
            "l_5meta",
            "r_toe",
            "l_toe",
            "r_pinky",
            "l_pinky",
            "r_index",
            "l_index",
            "l_eye",
            "r_eye",
            "r_ear",
            "l_ear",
            "nose",
            "L4",
            "T6",
            "r_big_toe",
            "l_big_toe",
        ],
        "coco_keypoints": [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ],
    }
]
