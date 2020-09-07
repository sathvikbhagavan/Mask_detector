import numpy as np
import argparse
import imutils
import time
import cv2
import keras


# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-f", "--face_detector", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-m", "--mask_detector", required=True, help="path to mask detection model")
args = vars(ap.parse_args())


# Load pre trained model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['face_detector'])


# Initialize the video stream from webcamera
print("[INFO] starting video stream...")
camera = cv2.VideoCapture(0)
time.sleep(2.0)
model = keras.models.load_model(args['mask_detector'])
color_dict = {
    0: (0, 255, 0),
    1: (0, 0, 255)
}

# Loop over the frames from the video stream
while True:
    _, frame = camera.read()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    label_rev_dict = {
        0: 'Mask',
        1: 'No_Mask'
    }
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype('int')
        img = frame[start_y:end_y, start_x:end_x]
        img_size = 100
        if img.size != 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_size, img_size))/255.0
            img = img.reshape(1, img.shape[0], img.shape[1], 1)
            result = model.predict(img)
            label = np.argmax(result, axis=1)[0]
        text = f"{label_rev_dict[label]}"
        y = start_y - 10 if start_y - 10 > 10 else start_y + 10
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color_dict[label], 2)
        cv2.putText(frame, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_dict[label], 2)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
