import cv2

# load images
image1 = cv2.imread("08-crop.jpg")
image2 = cv2.imread("08-normal.jpg")

Rimg1 = cv2.resize(image1, dsize=(255,255), interpolation=cv2.INTER_LINEAR)
Rimg2 = cv2.resize(image2, dsize=(255,255), interpolation=cv2.INTER_LINEAR)

# compute difference
difference = cv2.subtract(Rimg1, Rimg2)


# color the mask red
Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
difference[mask != 255] = [0, 0, 255]

# add the red mask to the images to make the differences obvious
Rimg1[mask != 255] = [0, 0, 255]
Rimg2[mask != 255] = [0, 0, 255]

# store images
cv2.imwrite('diffOverImage1.png', Rimg1)
cv2.imwrite('diffOverImage2.png', Rimg2)
cv2.imwrite('diff.png', difference)