import math
import glob
import cv2, time
import numpy as np
import dlib
from numpy import array, dot, mean, std, empty, argsort ,size, sqrt ,shape ,transpose, linalg

face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')


def generate_database(train_num,test_num):
    train_image_list = []
    train_label_list = []
    test_image_list = []
    test_label_list = []    
    count = 0
    scale_factor = 5.0

    for filename in glob.glob('./jpg/*.jpg'):

        im  = cv2.imread(filename)
        im  = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
        img = cv2.resize(im, (0,0), fx=1/scale_factor, fy=1/scale_factor)
        count = count + 1

        if count < train_num:
            train_image_list.append(img)
            if filename.find("fn") != -1:
                train_label_list.append(-1)
            else:
                train_label_list.append(1)
        else:
            test_image_list.append(img)
            if filename.find("fn") != -1:
                test_label_list.append(-1)
            else:
                test_label_list.append(1)
        
        if count == train_num+test_num:
            break;
        
    
    return train_image_list, train_label_list, test_image_list, test_label_list

 

def detect_face(img):
    face_mat = []
    num = 0
    for i in range(0,len(img)):
        rect = face_detector(img[i], 1)
        if len(rect) > 0:
           face_area = dlib.rectangle(rect[0].left(),rect[0].top(),rect[0].right(), rect[0].bottom())
           # shape = face_predictor(img[i], face_area)
           img_face = img[i][face_area.top():face_area.bottom(),face_area.left():face_area.right()]
           img_face = cv2.resize(img_face,(48,48))
           # cv2.imshow('image',img_face)
           # cv2.waitKey(0)
           h,w = img_face.shape[:2]  
           img_new = np.reshape(img_face, h*w) #flatten the 2d matrix to a row
           face_mat.append(img_new)
        else:
            print('face not find')
    return np.array(face_mat)


def pca(data):
    num, dim = data.shape[:2]
    mean_mat = data - data.mean(axis = 0)
    if num < dim:
        cov_mat = dot(mean_mat, mean_mat.T) # covariance matrix
        e_val,e_vec = linalg.eigh(cov_mat) # eigenvalues and eigenvectors
        tmp = dot(mean_mat.T,e_vec).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(e_val)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        U,S,V = linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data
    
    # return the projection matrix, the variance and the mean
    return V,S,mean_mat


time_start = time.clock()

train_num = 100
test_num = 20

[train_image, train_label, test_image,test_label] = generate_database(train_num,test_num)

train_data = detect_face(train_image)

test_data = detect_face(test_image)

[proj_matrix,variance,mean_mat] = pca(train_data)

eigen_face = proj_matrix * mean_mat

train_data = np.dot(eigen_face, train_data.T).T

time_end = time.clock()
print('frame time: {0}ms'.format(int((time_end - time_start) * 1000)))

#np.savetxt("eigenface.txt",eigenface)

