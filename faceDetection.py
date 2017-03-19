import math
import glob
import cv2, time
import numpy as np
import dlib
from numpy import array, dot, mean, std, empty, argsort ,size ,shape ,transpose
from numpy.linalg import eigh, solve

face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')


def generate_database():
    image_list = []
    label_list = []
    count = 0
    for filename in glob.glob('./jpg/*.jpg'):

        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
        img = cv2.resize(im, (0,0), fx=1/10.0, fy=1/10.0)
        h,w = img.shape[:2]
        img_new = np.reshape(img, h*w)
        image_list.append(img_new)
        count = count + 1

        if (count == 100):
            break
        if (filename.find("fn") != -1):
            label_list.append(-1)
        else:
            label_list.append(1)
    
        # cv2.imwrite(filename+"_bw",img)
    return image_list, label_list

 

def detect_face(img):
    face_mat = []
    for i in range(0,len(img)):
        rect = face_detector(img[i], 1)
        if len(rect) > 0:
           face_area = dlib.rectangle(rect[0].left(),rect[0].top(),rect[0].right(), rect[0].bottom())
           shape = face_predictor(img[i], face_area)
           face_mat.append(shape)
           # cv2.imshow('image',face_area)
           # cv2.waitKey(0)
        else:
            print('face not find')
    return face_mat


def pca(image, k, m):
    mean_mat = image - m
    C = np.dot(mean_mat.transpose(),mean_mat)
    E, V = eigh(C)                          #E eigen values and V is eigen vectors  
    key = argsort(E)[::-1][:k]                  #key will have indices of array
  
    E, V = E[key], V[:, key]
    V = V.T                                   #eigen matrix ka transpose bhj rha hu i.e. k*48
    eigenface = V*m
    return eigenface 



[image, label] = generate_database()
np.savetxt('data.txt', image)
np.savetxt('label.txt', label)
#data = detect_face(image)
m = mean(image, axis = 0)
k = 80
eigenface = pca(image,k,m)
np.savetxt("eigenface.txt",eigenface)

