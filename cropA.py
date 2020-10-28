from PIL import Image
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:/Users/user/AppData/Local/Programs/Python/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
src = cv2.imread("04.jpg")
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(src_gray,1.1,3)

for x, y, w, h in faces:
    cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face = src[y: y + h, x: x + w]
    face_gray = src_gray[y: y + h, x: x + w]
print(x,  y,  w,  h)


#이미지 크롭 코드
img1 = Image.open('04.jpg')
crop_area = (x,  y,  x+w,  y+h)
       ##(가로시작점, 세로시작점, 가로범위, 세로범위)
       ##보통 x, y w, h 변수로 사용됨

       ### 근데 저코드는 범위가 아니라 xy좌표 두개인듯
       ### 얼굴선은 따지는걸로 봐선 크롭만 xy 좌표 두개쓰는듯, 인식파트는 높이 넓이가 맞음
       #### 시작점은 그대로 2차좌표는 합치는걸로 수정함. -> 성공!
crop_img1 = img1.crop(crop_area)
crop_img1.save("041.jpg")
crop_img1.show()


face_cascade = cv2.CascadeClassifier('C:/Users/user/AppData/Local/Programs/Python/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
src = cv2.imread("08.jpg")
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(src_gray,1.1,3)

for x, y, w, h in faces:
    cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face = src[y: y + h, x: x + w]
    face_gray = src_gray[y: y + h, x: x + w]
print(x,  y,  w,  h)


#이미지 크롭 코드
img2 = Image.open('08.jpg')
crop_area = (x,  y,  x+w,  y+h)
       ##(가로시작점, 세로시작점, 가로범위, 세로범위)
       ##보통 x, y w, h 변수로 사용됨

       ### 근데 저코드는 범위가 아니라 xy좌표 두개인듯
       ### 얼굴선은 따지는걸로 봐선 크롭만 xy 좌표 두개쓰는듯, 인식파트는 높이 넓이가 맞음
       #### 시작점은 그대로 2차좌표는 합치는걸로 수정함. -> 성공!
crop_img2 = img2.crop(crop_area)
crop_img2.save("081.jpg")
crop_img2.show()


def addImage(imgfile, imgfile2):
    img1 = cv2.imread(imgfile)
    img2 = cv2.imread(imgfile2)


    img1 = cv2.resize(img1, (479, 557))
    img2 = cv2.resize(img2, (479, 557))

    print(img1.shape)
    print(img2.shape)

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)

    add_img1 = img1 + img2
    add_img2 = cv2.add(img1, img2)

    cv2.imshow("img1 + img2", add_img1)
    cv2.imshow("cv2.add", add_img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

addImage("041.jpg","081.jpg")
