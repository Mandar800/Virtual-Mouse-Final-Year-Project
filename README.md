# Virtual Mouse -Final Year Project
[![TensorFlow 1.15](https://img.shields.io/badge/TensorFlow-1.15-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![Protobuf Compiler >= 3.0](https://img.shields.io/badge/ProtoBuf%20Compiler-%3E3.0-brightgreen)](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager)


Software program which will Track/Monitor your Hand Movement in
Front of Screen through a Webcam and
will Move the Cursor of the Computing
System with respect to your hand
Movement and can Do certain Fixed
Tasks Like Right Click, Left Click double
left Click, Scroll and Movement of the
cursor.

### Additional Tools Used  
- [Pyautogui](https://pypi.org/project/PyAutoGUI/)
- [OpenCV](https://github.com/opencv/opencv)

### Neural Network Used
- Single Shot MultiBox Detector (SSD) 

### Features
- Cursor Movements
- Right Click
- Left Click
- Double Left Click
- Scroll

![](name-of-giphy.gif)

## Algorithm 
#### Step 1 - Object Detection by SSD Neural Network 
![](mouse.gif)
#### Step 2 - Calculating Center of Box from Box Co-ordinates 
#### Step 4 - Determining Desired Action From Box Center Location 
#### Step 5 - Performing Mouse Action on Different Thread for Different classes 
