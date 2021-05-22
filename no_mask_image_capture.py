import cv2
import cvlib as cv
import os

path = "./face_image/no_mask/"
face_list = os.listdir(path)
img_num = 0
for face_img in face_list:
    print(face_img)
    path1 = path + face_img
    img = cv2.imread(path1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, confidence = cv.detect_face(img)
    print(confidence)
    for i, f in enumerate(faces):
        print(f)
        cropped = img[f[1]:f[3], f[0]:f[2]]
        file_name = "./nomask/" + str(img_num) + ".jpg"
        try:
            cv2.imwrite(file_name, cropped)
            img_num = img_num + 1
        except:
            continue
        img_num = img_num + 1


