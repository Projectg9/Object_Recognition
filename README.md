Real Time Object Recognition from Video

Object recognition is a process for identifying a specific object in a digital image or video. Object recognition algorithms rely on
matching, learning, or pattern recognition algorithms using appearance-based or feature-based techniques. Object recognition is
useful in applications such as video stabilization, advanced driver assistance systems (ADAS), and disease identification in bioimaging.
2 Common techniques include deep learning based approaches such as convolutional neural networks, and feature-based approaches using
edges, gradients, histogram of oriented gradients (HOG).
                          It is also used in tracking objects, for example tracking a ball during a
football match, tracking movement of a cricket bat, This task is still a challenge for computer vision systems.
2 Every object class has its own special features that helps in classifying the class for example all circles are round. For example, when looking for circles, objects that are at a particular distance from a point are sought. Similarly, when looking for squares, objects that are perpendicular at corners and have equal side lengths are needed.
Features

Recognition of object from the image

An Image is given as input, which contains a set of objects:



import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_locations = face_recognition.face_locations(image)
Find and manipulate facial features in pictures

Get the locations and outlines of each person's eyes, nose, mouth and chin.



import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)


Installation

Requirements

Python 3.3+ or Python 2.7
Linux)
Installation Options:

Installing  Linux

First, make sure you have dlib already installed with Python bindings:

How to install dlib from source on macOS or Ubuntu
Then, install this module from pypi using pip3 (or pip2 for Python 2):

pip3 install face_recognition
If you are having trouble with installation, you can also try out a pre-configured VM.



