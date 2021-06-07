# import necessary packages
import cvlib as cv
import cv2
import numpy as np
from keras.applications.efficientnet import preprocess_input
from keras_preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model

model = load_model('mask_detector.h5')
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    face_num, confi = cv.detect_face(frame)
    for i, f in enumerate(face_num):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        face = frame[startY:endY, startX:endX]
        face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
        x = img_to_array(face)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        confidence = model.predict(x)
        if confidence < 0.5:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            text = "NO MASK!!!"
            cv2.putText(frame, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            text = "OK"
            cv2.putText(frame, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        print(confidence)
    cv2.imshow("Mask Detector", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
