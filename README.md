# Mask Detector

Mask Detector - detects your face and classifies whether you are wearing a face mask or not streaming through webcam

This project uses a pretrained Caffemodel for face recognition and a Conv Net Classifier for classifying mask or non mask

Required Modules : OpenCV, Tensorflow(>2.0), Keras

Command line Arguements required : 

--prototxt : path to Caffe 'deploy' prototxt file

--mask_detector : path to mask detection model

--face_detector : path to Caffe pre-trained model

Example run would be : python mask_detector_video.py --prototxt deploy.prototxt.txt --mask_detector mask_detector.h5 --face_detector face.caffemodel
