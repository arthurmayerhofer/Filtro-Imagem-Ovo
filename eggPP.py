import cv2 

img = cv2.imread('01.jpg')

rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray1 = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

limiar, bi = cv2.threshold (img, 0, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('image',gray1)
cv2.imshow('image2',gray2)


cv2.waitKey(0)
cv2.destroyAllWindows()