import cv2
import cvlib as cv

cap = cv2.VideoCapture(0)
img_num = 0
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_num, confi = cv.detect_face(frame)
    for i, f in enumerate(face_num):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        face_img = frame[startY:endY, startX:endX, :]
        cv2.imwrite('./mask/Yes' + str(img_num) + '.jpg', face_img)
        img_num = img_num + 1
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()