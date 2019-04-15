from tqdm import tqdm
from keras.utils import to_categorical
from random import shuffle
import cv2
import numpy as np
from bone_info import *
IMG_SIZE = 224
TRAIN_DATA = []
TEST_DATA = []


def create_data(train_data, path, bone_number):
    for item in tqdm(os.listdir(path)):
        patient_path = os.path.join(path, item)
        for patient_study in os.listdir(patient_path):
            p_path = os.path.join(patient_path, patient_study)
            label = to_categorical(bone_number, 7)
            for patient_image in os.listdir(p_path):
                image_path = os.path.join(p_path, patient_image)
                bgr = cv2.imread(image_path)
                if bgr is None:
                    continue
                enhance_img = img_hist_eq(bgr)
                img = cv2.resize(enhance_img, (IMG_SIZE, IMG_SIZE))
                img = np.divide(img, 255)
                train_data.append([np.array(img), label])
    shuffle(train_data)
    return train_data


def img_hist_eq(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


create_data(TRAIN_DATA, ELBOW_TRAIN_PATH, ELBOW)
print("SIZE OF TRAIN DATA IS :", len(TRAIN_DATA))

create_data(TRAIN_DATA, SHOULDER_TRAIN_PATH, SHOULDER)
print("SIZE OF TRAIN DATA IS :", len(TRAIN_DATA))

create_data(TRAIN_DATA, FOREARM_TRAIN_PATH, FOREARM)
print("SIZE OF TRAIN DATA IS :", len(TRAIN_DATA))

create_data(TRAIN_DATA, HAND_TRAIN_PATH, HAND)
print("SIZE OF TRAIN DATA IS :", len(TRAIN_DATA))

create_data(TRAIN_DATA, WRIST_TRAIN_PATH, WRIST)
print("SIZE OF TRAIN DATA IS :", len(TRAIN_DATA))

create_data(TRAIN_DATA, FINGER_TRAIN_PATH, FINGER)
print("SIZE OF TRAIN DATA IS :", len(TRAIN_DATA))

create_data(TRAIN_DATA, HUMERUS_TRAIN_PATH, HUMERUS)
print("SIZE OF TRAIN DATA IS :", len(TRAIN_DATA))
print("All Train Data Readed!!")



# ************************************************************************************

create_data(TEST_DATA, ELBOW_TEST_PATH, ELBOW)
print("SIZE OF TEST DATA IS :", len(TEST_DATA))

create_data(TEST_DATA, SHOULDER_TEST_PATH, SHOULDER)
print("SIZE OF TEST DATA IS :", len(TEST_DATA))

create_data(TEST_DATA, FOREARM_TEST_PATH, FOREARM)
print("SIZE OF TEST DATA IS :", len(TEST_DATA))

create_data(TEST_DATA, HAND_TEST_PATH, HAND)
print("SIZE OF TEST DATA IS :", len(TEST_DATA))

create_data(TEST_DATA, WRIST_TEST_PATH, WRIST)
print("SIZE OF TEST DATA IS :", len(TEST_DATA))

create_data(TEST_DATA, FINGER_TEST_PATH, FINGER)
print("SIZE OF TEST DATA IS :", len(TEST_DATA))

create_data(TEST_DATA, HUMERUS_TEST_PATH, HUMERUS)
print("SIZE OF TEST DATA IS :", len(TEST_DATA))
print("All Test Data Readed!!")


X = np.array([i[0] for i in TRAIN_DATA]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
print("TRAIN SHAPE", X.shape)
y = np.array([i[1] for i in TRAIN_DATA])


x_test = np.array([i[0] for i in TEST_DATA]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
print("TEST SHAPE", x_test.shape)
y_test = np.array([i[1] for i in TEST_DATA])
print(y_test.shape)