import numpy as np
import argparse
import imutils
import time
import cv2
import keras


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-f", "--face_detector", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-m", "--mask_detector", required=True,
                help="path to mask detection model")
args = vars(ap.parse_args())


# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['face_detector'])


# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)
model = keras.models.load_model(args['mask_detector'])
color_dict = {
    0: (0, 255, 0),
    1: (0, 0, 255)
}

# loop over the frames from the video stream
while True:
    _, frame = vs.read()
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
        if confidence < args['confidence']:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        img = frame[startY:endY, startX:endX]
        img_size = 100
        if img.size != 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_size, img_size))/255.0
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=-1)
            result = model.predict(img)
            label = np.argmax(result, axis=1)[0]
        text = f"{label_rev_dict[label]}"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), color_dict[label], 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_dict[label], 2)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
