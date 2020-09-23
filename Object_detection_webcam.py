######## Webcam Object Detection Using Tensorflow-trained Classifier ###########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a webcam feed.
# It draws boxes, scores, and labels around the objects of interest in each frame
# from the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me and my Application.

################################################################################


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pyautogui, sys
import time
import threading


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
pyautogui.FAILSAFE = False


# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'


# Grab path to current working directory
CWD_PATH = os.getcwd()


# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')


# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')


# Number of classes the object detector can identify
NUM_CLASSES = 5


## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier


# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')


# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')


# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)


#Threading code for MOuse movement
index = 0
def Mouse():
    global index,posx,posy
    speed=1 # Speed Factor of the cursor
    while True:
        if index == 0: #do Nothing and restore speed 
            speed=1
            pass
            
        elif index == 1 :#right top
            #print(1)
            pyautogui.moveTo(posx+10*speed, posy-10*speed,0.1)
            #time.sleep(0.005)
            
        
        elif index == 2 :#right
            #print(2)
            pyautogui.moveTo(posx+10*speed, posy,0.1)
            #time.sleep(0.002)
            
        
        elif index == 3 :#right bottom
            #print(3)
            pyautogui.moveTo(posx+10*speed, posy+10*speed,0.1)
            #time.sleep(0.005)
            #index=0
        
        elif index == 4 :#bottom
            #print(4)
            pyautogui.moveTo(posx, posy+10*speed,0.1)
            #time.sleep(0.005)
            #index=0
        
        elif index == 5 :#left bottom
            #print(5)
            pyautogui.moveTo(posx-10*speed, posy+10*speed,0.1)
            #time.sleep(0.005)
            #index=0
        
        elif index == 6 :#left
            #print(6)
            pyautogui.moveTo(posx-10*speed, posy,0.1)
            #time.sleep(0.005)
            #index=0
           
        elif index == 7 :#left top
            #print(7)
            pyautogui.moveTo(posx-10*speed, posy-10*speed,0.1)
            #time.sleep(0.005)
            #index=0
        
        elif index == 8 :#top
            #print(8)
            pyautogui.moveTo(posx, posy-10*speed,0.1)
            #time.sleep(0.005)
            #index=0
        
        elif index==9:
            break
        speed+=0.01

# Start Thread 
T = threading.Thread(target=Mouse)
T.start()


while(True):
    
    #start = time.clock()
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame = cv2.flip(frame,1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    frame_expanded = np.expand_dims(frame_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # get co-ordinates of firsst box detected
    top  = int(boxes[0][0][0]*720)
    bottom = int(boxes[0][0][2]*720)
    left = int(boxes[0][0][1]*1280)
    right = int(boxes[0][0][3]*1280)

    # center of the box 
    y = (top+bottom)//2
    x = (left+right)//2
    
    #print(x,y)

    # defining safe area / no action area / rest area 
    ltopx = int(0.50*1280)
    ltopy = int(0.35*720)
    rbotx = int(0.75*1280)
    rboty = int(0.65*720)
    
    cv2.rectangle(frame,(ltopx,ltopy),(rbotx,rboty),(0,255,0),3)

    #get Scores and class detected
    scr = int(scores[0][0]*100)
    action = int(classes[0][0])
    
    #print(scr, action)
    
    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60,
        max_boxes_to_draw=1)
    cv2.circle(frame,(x,y),10,(0,0,255),-1)
    
    #print(T.isAlive(), index)
    
    # Mouse Movement
    posx,posy = pyautogui.position() # get Current position
    
    if x>ltopx and x<rbotx and y<rboty and y>ltopy: # if inside center area 
        index=0
    
    if action==1 and scr>40 :
        
        if x>=rbotx and y<ltopy : # top right
            index=1
            
        if x>=rbotx and y>=ltopy and y<rboty :# right
            index=2
            
        if x>=rbotx and y>rboty : # bottom right
            index=3
            
        if x>=ltopx and x<=rbotx and y>rboty : # bottom
            index=4
            
        if x<=ltopx and y>rboty :# left botom
            index=5
            
        if x<=ltopx and y>=ltopy and y<rboty : # left
            index=6
            
        if x<=ltopx and y<ltopy : # left top
            index=7
            
        if x>=ltopx and x<=rbotx and y<ltopy : # top
            index=8
            
    elif action==2 and scr>98: # click
        pyautogui.click()
        
    elif action==3 and scr>98: # Double click
        pyautogui.doubleClick()
        
    elif action==4 and scr>98: # right click
        pyautogui.rightClick()
        
    elif action==5 and scr>98: # scroll
        pyautogui.scroll(-10)
    else:
        index=0
  
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    
    #end=time.clock()
    #print(end-start)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        index=9
        break
        

    
# Clean up
video.release()
cv2.destroyAllWindows()

