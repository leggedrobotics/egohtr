from src.retargeter.retarget_helper import smplx_to_egomc_map

# (root + 55)
SMPLX_BLENDER_JOINTS = [
    'root', 'pelvis', 'left_hip', 'left_knee', 'left_ankle', 'left_foot',
    'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'spine1', 'spine2', 
    'spine3', 'neck', 'head', 'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 
    'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_index1', 'left_index2', 
    'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 
    'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 
    'left_thumb3', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_index1',
    'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 
    'right_pinky2', 'right_pinky3','right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1',
    'right_thumb2', 'right_thumb3'
]

# rokoko mocap suit (61)
SMPLX_ROKOKO_JOINTS = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftForeArmRoll', 'LeftArmRoll', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightForeArmRoll', 'RightArmRoll', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftLegRoll', 'LeftUpLegRoll', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'
]

# qualisys mocap suit (64)
SMPLX_QUALISYS_JOINTS = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftForeArmRoll', 'LeftHand', 'LeftInHandThumb', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftInHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftInHandMiddle', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftInHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftInHandPinky', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightForeArmRoll', 'RightHand', 'RightInHandThumb', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightInHandIndex', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightInHandMiddle', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightInHandRing', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightInHandPinky', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']

# base SMPL joints (24)
SMPL_JOINTS = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand'
]


# base SMPLX body joints (21)
SMPLX_BODY_JOINTS = [
    'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
]

# SMPLX model joints (127 + face contour)
SMPLX_JOINT_NAMES = [
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

def get_retarget_idxs(mocap_type: str) -> list[int]:
    """ Get list of EgoMC joint indices corresponding to SMPLX joints (first 22 joints) """
    retarget_idxs = []
    smplx_idxs = []
    joint_list = None
    target_joints = ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle', 'left_knee', 'right_knee']

    if mocap_type == 'egomc':
        joint_list = SMPLX_ROKOKO_JOINTS
    elif mocap_type == 'exomc':
        joint_list = SMPLX_QUALISYS_JOINTS


    for smplx_joint in target_joints:
        egomc_joint = smplx_to_egomc_map[smplx_joint]
        egomc_idx = joint_list.index(egomc_joint)
        retarget_idxs.append(egomc_idx)
        smplx_idxs.append(SMPLX_JOINT_NAMES.index(smplx_joint))

    return retarget_idxs, smplx_idxs
