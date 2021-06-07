import cv2
import cvlib as cv
import os

mask_list = os.listdir("./face_image/mask2")
img_num = 0
path = "./face_image/mask2/"
for mask_img in mask_list:
    print(mask_img)
    img = cv2.imread(path + mask_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, confidence = cv.detect_face(img)
    print(confidence)
    for i, f in enumerate(faces):
        print(f)
        cropped = img[f[1]:f[3], f[0]:f[2]]
        file_name = "./mask2/YES" + str(img_num) + ".jpg"
        try:
            cv2.imwrite(file_name, cropped)
            img_num = img_num + 1
        except:
            continue


