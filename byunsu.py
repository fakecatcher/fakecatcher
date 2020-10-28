from PIL import Image
import numpy as np
import cv2

#1) a추출
fullimg = '04.jpg'
faceimg = '04-crop.jpg'

face_cascade = cv2.CascadeClassifier(
    'C:/Users/user/AppData/Local/Programs/Python/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
src = cv2.imread(fullimg)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(src_gray, 1.1, 3)

for x, y, w, h in faces:
    cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face = src[y: y + h, x: x + w]
    face_gray = src_gray[y: y + h, x: x + w]
print(x, y, w, h)

# 이미지 크롭 코드
img1 = Image.open(fullimg)
crop_area = (x, y, x + w, y + h)
    ##(가로시작점, 세로시작점, 가로범위, 세로범위)
    ##보통 x, y w, h 변수로 사용됨

    ### 근데 저코드는 범위가 아니라 xy좌표 두개인듯((x,y), (w,h) )
    ### 얼굴선은 따지는걸로 봐선 크롭만 xy 좌표 두개쓰는듯, 인식파트는 높이 넓이가 맞음
    #### 시작점은 그대로 2차좌표는 합치는걸로 수정함. -> 성공!
crop_img1 = img1.crop(crop_area)
crop_img1.save(faceimg)


#2)
fullimg = '08.jpg'
faceimg = '08-crop.jpg'

face_cascade = cv2.CascadeClassifier(
    'C:/Users/user/AppData/Local/Programs/Python/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
src = cv2.imread(fullimg)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(src_gray, 1.1, 3)

for x, y, w, h in faces:
    cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face = src[y: y + h, x: x + w]
    face_gray = src_gray[y: y + h, x: x + w]
print(x, y, w, h)

# 이미지 크롭 코드
img2 = Image.open(fullimg)
crop_area = (x, y, x + w, y + h)
crop_img2 = img2.crop(crop_area)
crop_img2.save(faceimg)

# 반드시 detec과 crop은 연달아 사용할것
# 두 전역함수의 fullimg 파일명은 동일히 쓸것.
# fullimg는 원본파일명
# faceimg는 새로 저장될 크롭이미지 (파일명 직접지정요망)
# addimg 함수는 이전과 동일

src[x:x + w, y:y + h] = crop_img1
cv2.imshow('src',src)
