import numpy as np

# base SMPLX body joints (21)
SMPLX_BODY_JOINTS = [
    'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
]

# SMPLH model joints (52)
SMPLH_JOINT_NAMES = [
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
]

SMPLH_SKELETON = np.array([
    [ 1,  0],  # left_hip   <- pelvis
    [ 2,  0],  # right_hip  <- pelvis
    [ 3,  0],  # spine1     <- pelvis

    [ 4,  1],  # left_knee   <- left_hip
    [ 5,  2],  # right_knee  <- right_hip
    [ 6,  3],  # spine2      <- spine1
    [ 7,  4],  # left_ankle  <- left_knee
    [ 8,  5],  # right_ankle <- right_knee
    [ 9,  6],  # spine3      <- spine2
    [10,  7],  # left_foot   <- left_ankle
    [11,  8],  # right_foot  <- right_ankle
    [12,  9],  # neck        <- spine3
    [13,  9],  # left_collar  <- spine3
    [14,  9],  # right_collar <- spine3
    [15, 12],  # head         <- neck
    [16, 13],  # left_shoulder  <- left_collar
    [17, 14],  # right_shoulder <- right_collar
    [18, 16],  # left_elbow     <- left_shoulder
    [19, 17],  # right_elbow    <- right_shoulder
    [20, 18],  # left_wrist     <- left_elbow
    [21, 19],  # right_wrist    <- right_elbow

    [22, 20],  # left_index1  <- left_wrist
    [23, 22],  # left_index2  <- left_index1
    [24, 23],  # left_index3  <- left_index2
    [25, 20],  # left_middle1 <- left_wrist
    [26, 25],  # left_middle2 <- left_middle1
    [27, 26],  # left_middle3 <- left_middle2
    [28, 20],  # left_pinky1  <- left_wrist
    [29, 28],  # left_pinky2  <- left_pinky1
    [30, 29],  # left_pinky3  <- left_pinky2
    [31, 20],  # left_ring1   <- left_wrist
    [32, 31],  # left_ring2   <- left_ring1
    [33, 32],  # left_ring3   <- left_ring2
    [34, 20],  # left_thumb1  <- left_wrist
    [35, 34],  # left_thumb2  <- left_thumb1
    [36, 35],  # left_thumb3  <- left_thumb2

    [37, 21],  # right_index1  <- right_wrist
    [38, 37],  # right_index2  <- right_index1
    [39, 38],  # right_index3  <- right_index2
    [40, 21],  # right_middle1 <- right_wrist
    [41, 40],  # right_middle2 <- right_middle1
    [42, 41],  # right_middle3 <- right_middle2
    [43, 21],  # right_pinky1  <- right_wrist
    [44, 43],  # right_pinky2  <- right_pinky1
    [45, 44],  # right_pinky3  <- right_pinky2
    [46, 21],  # right_ring1   <- right_wrist
    [47, 46],  # right_ring2   <- right_ring1
    [48, 47],  # right_ring3   <- right_ring2
    [49, 21],  # right_thumb1  <- right_wrist
    [50, 49],  # right_thumb2  <- right_thumb1
    [51, 50],  # right_thumb3  <- right_thumb2
], dtype=np.int32)

# SMPLX model joints (127)
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

HEAD_JOINT_NAMES = [
    "neck",
    "head",
   #  "jaw",
   #  "left_eye_smplhf",
   #  "right_eye_smplhf",
]

TRUNK_JOINT_NAMES = [
    "pelvis",
    "spine1",
    "spine2",
    "spine3",
]

UPPER_LIMBS_JOINT_NAMES = [
    "left_collar",
    "right_collar",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

LOWER_LIMBS_JOINT_NAMES = [
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_foot",
    "right_foot",
]


EVAL_JOINT_NAMES = [
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
]

CONTACT_CANDIDATES = {
    'RightFoot': np.array([
       8459, 8460, 8461, 8462, 8463, 8464, 8465, 8466, 8467, 8468, 8469,
       8470, 8471, 8472, 8473, 8474, 8475, 8476, 8477, 8478, 8479, 8480,
       8481, 8482, 8483, 8484, 8485, 8486, 8487, 8488, 8489, 8490, 8491,
       8492, 8493, 8494, 8495, 8496, 8497, 8498, 8499, 8500, 8501, 8502,
       8503, 8504, 8505, 8506, 8507, 8508, 8509, 8510, 8511, 8512, 8513,
       8514, 8515, 8516, 8517, 8518, 8519, 8520, 8521, 8522, 8523, 8524,
       8525, 8526, 8527, 8528, 8529, 8530, 8531, 8532, 8533, 8534, 8535,
       8536, 8537, 8538, 8540, 8541, 8542, 8543, 8544, 8545, 8546, 8547,
       8548, 8549, 8550, 8551, 8552, 8553, 8554, 8555, 8556, 8557, 8558,
       8559, 8560, 8562, 8563, 8564, 8565, 8566, 8596, 8597, 8598, 8599,
       8600, 8601, 8602, 8603, 8604, 8605, 8606, 8607, 8608, 8609, 8610,
       8611, 8646,
       8654, 8655, 8656, 8667, 8668, 8669, 8670, 8671, 8672, 8673, 8674,
       8675, 8676, 8677, 8678, 8679, 8685, 8686, 8687, 8688, 8689, 8690,
       8691, 8692, 8693, 8694, 8695, 8696, 8697, 8698, 8699, 8700, 8701,
       8702, 8703, 8704, 8705, 8706, 8707, 8708, 8709, 8710, 8711, 8712,
       8713, 8714, 8715, 8716]), 
    'LeftFoot': np.array([
       5765, 5767, 5768, 5771, 5772, 5773, 5774, 5775, 5776, 5777, 5778,
       5779, 5780, 5781, 5782, 5783, 5784, 5789, 5790, 5791, 5792, 5793,
       5794, 5795, 5796, 5797, 5798, 5803, 5804, 5805, 5806, 5807, 5808,
       5813, 5814, 5815, 5816, 5817, 5818, 5819, 5820, 5823, 5824, 5827,
       5828, 5829, 5830, 5831, 5832, 5833, 5836, 5839, 5840, 5841, 5842,
       5843, 5844, 5847, 5848, 5849, 5850, 5851, 5852, 5853, 5854, 5855,
       5856, 5859, 5860, 5861, 5862, 5863, 5864, 5865, 5866, 5868, 5869,
       5870, 5903, 5904, 5905, 5906, 5907, 5908, 5909, 5910, 5911, 5912,
       5913, 5914, 5915, 5916, 5917, 8866, 8897, 8898, 
       8849, 8856, 8867, 8868, 8874, 8879, 8880, 8881, 8882, 8883, 8884,
       8885, 8886, 8887, 8888, 8889, 8890, 8891, 8899, 8900, 8901, 8902,
       8903, 8904, 8905, 8906, 8907, 8908, 8909, 8910, 8911, 8912, 8913,
       8914, 8915, 8916, 8917, 8918, 8919, 8920, 8921, 8922, 8923, 8924,
       8925, 8929, 8930, 8934]), 
   }

CONTACT_VERTEX_CANDIDATES = {
    'RightFront': np.array([8459, 8460, 8461, 8462, 8463, 8464, 8465, 8466, 8467, 8468, 8469,
       8470, 8471, 8472, 8473, 8474, 8475, 8476, 8477, 8478, 8479, 8480,
       8481, 8482, 8483, 8484, 8485, 8486, 8487, 8488, 8489, 8490, 8491,
       8492, 8493, 8494, 8495, 8496, 8497, 8498, 8499, 8500, 8501, 8502,
       8503, 8504, 8505, 8506, 8507, 8508, 8509, 8510, 8511, 8512, 8513,
       8514, 8515, 8516, 8517, 8518, 8519, 8520, 8521, 8522, 8523, 8524,
       8525, 8526, 8527, 8528, 8529, 8530, 8531, 8532, 8533, 8534, 8535,
       8536, 8537, 8538, 8540, 8541, 8542, 8543, 8544, 8545, 8546, 8547,
       8548, 8549, 8550, 8551, 8552, 8553, 8554, 8555, 8556, 8557, 8558,
       8559, 8560, 8562, 8563, 8564, 8565, 8566, 8596, 8597, 8598, 8599,
       8600, 8601, 8602, 8603, 8604, 8605, 8606, 8607, 8608, 8609, 8610,
       8611, 8646]),
    'RightHeel': np.array([8654, 8655, 8656, 8667, 8668, 8669, 8670, 8671, 8672, 8673, 8674,
       8675, 8676, 8677, 8678, 8679, 8685, 8686, 8687, 8688, 8689, 8690,
       8691, 8692, 8693, 8694, 8695, 8696, 8697, 8698, 8699, 8700, 8701,
       8702, 8703, 8704, 8705, 8706, 8707, 8708, 8709, 8710, 8711, 8712,
       8713, 8714, 8715, 8716]), 
    'LeftFront': np.array([5765, 5767, 5768, 5771, 5772, 5773, 5774, 5775, 5776, 5777, 5778,
       5779, 5780, 5781, 5782, 5783, 5784, 5789, 5790, 5791, 5792, 5793,
       5794, 5795, 5796, 5797, 5798, 5803, 5804, 5805, 5806, 5807, 5808,
       5813, 5814, 5815, 5816, 5817, 5818, 5819, 5820, 5823, 5824, 5827,
       5828, 5829, 5830, 5831, 5832, 5833, 5836, 5839, 5840, 5841, 5842,
       5843, 5844, 5847, 5848, 5849, 5850, 5851, 5852, 5853, 5854, 5855,
       5856, 5859, 5860, 5861, 5862, 5863, 5864, 5865, 5866, 5868, 5869,
       5870, 5903, 5904, 5905, 5906, 5907, 5908, 5909, 5910, 5911, 5912,
       5913, 5914, 5915, 5916, 5917, 8866, 8897, 8898]), 
    'LeftHeel': np.array([8849, 8856, 8867, 8868, 8874, 8879, 8880, 8881, 8882, 8883, 8884,
       8885, 8886, 8887, 8888, 8889, 8890, 8891, 8899, 8900, 8901, 8902,
       8903, 8904, 8905, 8906, 8907, 8908, 8909, 8910, 8911, 8912, 8913,
       8914, 8915, 8916, 8917, 8918, 8919, 8920, 8921, 8922, 8923, 8924,
       8925, 8929, 8930, 8934]), 
   }



joint_names_in_order = [ #bvh order
   'root', 'pelvis', 'left_hip', 'left_knee', 'left_ankle', 'left_foot',
   'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'spine1',
   'spine2', 'spine3', 'neck', 'head', 'jaw', 'left_eye_smplhf',
   'right_eye_smplhf', 'left_collar', 'left_shoulder', 'left_elbow',
   'left_wrist', 'left_index1', 'left_index2', 'left_index3',
   'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1',
   'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3',
   'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_collar',
   'right_shoulder', 'right_elbow', 'right_wrist', 'right_index1',
   'right_index2', 'right_index3', 'right_middle1', 'right_middle2',
   'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3',
   'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1',
   'right_thumb2', 'right_thumb3'
]

# SMPL-X parameter mappings
root_joint_name = 'pelvis'
body_pose_joints = [
   'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
   'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
   'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
   'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
]
lhand_pose_joints = [
   'left_index1', 'left_index2', 'left_index3', 'left_middle1',
   'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2',
   'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3',
   'left_thumb1', 'left_thumb2', 'left_thumb3'
]
rhand_pose_joints = [
   'right_index1', 'right_index2', 'right_index3', 'right_middle1',
   'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2',
   'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3',
   'right_thumb1', 'right_thumb2', 'right_thumb3'
]
jaw_pose_joint = 'jaw'
leye_pose_joint = 'left_eye_smplhf'
reye_pose_joint = 'right_eye_smplhf'

