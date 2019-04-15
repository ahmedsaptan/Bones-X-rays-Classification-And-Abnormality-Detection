import os

ELBOW = 0
FINGER = 1
FOREARM = 2
HAND = 3
HUMERUS = 4
SHOULDER = 5
WRIST = 6


rootdir = os.getcwd() + "/MURA-v1.1"

ELBOW_TRAIN_PATH = rootdir + "/train/XR_ELBOW"
FINGER_TRAIN_PATH = rootdir + "/train/XR_FINGER"
HAND_TRAIN_PATH = rootdir + "/train/XR_HAND"
SHOULDER_TRAIN_PATH = rootdir + "/train/XR_SHOULDER"
WRIST_TRAIN_PATH = rootdir + "/train/XR_WRIST"
HUMERUS_TRAIN_PATH = rootdir + "/train/XR_HUMERUS"
FOREARM_TRAIN_PATH = rootdir + "/train/XR_FOREARM"

ELBOW_TEST_PATH = rootdir + "/valid/XR_ELBOW"
FINGER_TEST_PATH = rootdir + "/valid/XR_FINGER"
HAND_TEST_PATH = rootdir + "/valid/XR_HAND"
SHOULDER_TEST_PATH = rootdir + "/valid/XR_SHOULDER"
WRIST_TEST_PATH = rootdir + "/valid/XR_WRIST"
HUMERUS_TEST_PATH = rootdir + "/valid/XR_HUMERUS"
FOREARM_TEST_PATH = rootdir + "/valid/XR_FOREARM"