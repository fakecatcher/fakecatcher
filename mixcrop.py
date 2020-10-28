# SeamlessClone을 활용한 이미지 합성 (seamlessclone.py)
from PIL import Image
import cv2
import numpy as np
import matplotlib.pylab as plt

# --① 합성 대상 영상 읽기
img1 = cv2.imread("081.jpg") #small
img2 = cv2.imread("041.jpg") #big
Rimg1 = cv2.resize(img1, dsize=(500,500), interpolation=cv2.INTER_LINEAR)
Rimg2 = cv2.resize(img2, dsize=(500,500), interpolation=cv2.INTER_LINEAR)
# --② 마스크 생성, 합성할 이미지 전체 영역을 255로 셋팅
mask = np.full_like(Rimg1, 255)

# --③ 합성 대상 좌표 계산(img2의 중앙)
height, width = Rimg2.shape[:2]
center = (width // 2, height // 2)

# --④ seamlessClone 으로 합성
normal = cv2.seamlessClone(Rimg1, Rimg2, mask, center, cv2.NORMAL_CLONE)
mixed = cv2.seamlessClone(Rimg1, Rimg2, mask, center, cv2.MIXED_CLONE)

# --⑤ 결과 출력
cv2.imshow('normal', normal)
cv2.imshow('mixed', mixed)
cv2.waitKey()
cv2.destroyAllWindows()