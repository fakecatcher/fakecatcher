import numpy as np
import cv2

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

addImage("croptest1233.jpg","test2.jpg")
