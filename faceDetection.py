import math
import cv2, time
import numpy as np
import dlib
import skimage
import skimage.transform


face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')



def find_landmark(img):
    #resize the img to 0.1 for face detection to speed up for real time
    scale_factor = 8
    resized_img = cv2.resize(img, (0,0), fx=1.0/scale_factor, fy=1.0/scale_factor);
    rect = face_detector(resized_img, 1)
    if len(rect) > 0:
    	face_area = dlib.rectangle(rect[0].left()*scale_factor, rect[0].top()*scale_factor, \
            rect[0].right()*scale_factor, rect[0].bottom()*scale_factor)
    	shape = face_predictor(img, face_area)
    	return shape
    else:
        print('face not find')
	return None
