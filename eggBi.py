import cv2 
import numpy as np
from matplotlib import pyplot as plt

kernelA = np.ones((2,2),np.uint8)
kernelB = np.ones((3,3),np.uint8)

kernelA = np.ones((2,2),np.uint8)
kernelB = np.ones((3,3),np.uint8)

bgr = cv2.imread('01.jpg',0)
rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(img,5)

ret,thr = cv2.threshold(blur,155,255,cv2.THRESH_BINARY)

adapM = cv2.adaptiveThreshold(thr,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

adapG = cv2.adaptiveThreshold(thr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
          cv2.THRESH_BINARY,11,2)

laplacian = cv2.Laplacian(img,cv2.CV_8U)

blFilter = cv2.bilateralFilter(laplacian, 9, 75, 75)

gaussL = cv2.GaussianBlur(blFilter, (5,5), 0)


sobelx = cv2.Sobel(laplacian,cv2.CV_8UC1,1,0,ksize=5)
sobelx = cv2.morphologyEx(sobelx, cv2.MORPH_CLOSE, kernelA)
sobelx = cv2.morphologyEx(sobelx, cv2.MORPH_OPEN, kernelA)
sobelx = cv2.dilate(sobelx, kernelB, iterations = 2)
sobelx = cv2.GaussianBlur(blFilter, (5,5), 0)

sobely = cv2.Sobel(laplacian,cv2.CV_8UC1,0,1,ksize=5)
sobely = cv2.morphologyEx(sobely, cv2.MORPH_CLOSE, kernelA)
sobely = cv2.morphologyEx(sobely, cv2.MORPH_OPEN, kernelA)
sobely = cv2.dilate(sobely, kernelB, iterations = 2)
sobely = cv2.GaussianBlur(blFilter, (5,5), 0)
elementoEstruturante = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
sobely = cv2.erode(sobely, elementoEstruturante, iterations = 2)
sobely = cv2.cvtColor(sobely,cv2.COLOR_GRAY2RGB) 
hsv = cv2.cvtColor(sobely,cv2.COLOR_RGB2HSV) 



sobelyR = cv2.Sobel(laplacian,cv2.CV_8UC1,0,1,ksize=5)
sobelyR = cv2.dilate(sobelyR, kernelB, iterations = 2)
sobelyR = cv2.morphologyEx(sobelyR, cv2.MORPH_CLOSE, kernelA)
sobelyR= cv2.morphologyEx(sobelyR, cv2.MORPH_OPEN, kernelA)
sobelyR = cv2.GaussianBlur(blFilter, (5,5), 0)
sobelyR = cv2.morphologyEx(sobelyR, cv2.MORPH_TOPHAT, 
	cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25)))
#sobelyR = cv2.bilateralFilter(sobelyR, 9, 75, 75)
#canny = cv2.subtract(sobely, gaussL)
'''
lower_black = np.array([0,0,0], dtype = "uint8")
upper_black = np.array([70,70,70], dtype = "uint8")
black_mask = cv2.inRange(hsv, lower_black, upper_black)

cv2.imshow('mask1',black_mask)
'''
x,y = sobely[0], sobely[1]

for linha in range(len(sobely)):
		for coluna in range(len(sobely[linha])):
			lista = sobely [linha][coluna].tolist()
			if linha > 0 and linha < 20 :
				sobely[x,y] = 0
			else:
				sobely[x,y] = 255


contours, hierarchy = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
print("Found %d objects." % len(contours))
for (i, c) in enumerate(contours):
	print("\tSize of contour %d: %d" % (i, len(c)))

shape = rgb.copy()
cv2.drawContours(shape, contours, -1, (0, 255, 0), 2)
cv2.imshow("Edges", shape)


images = [hsv, 0, sobelx,
          laplacian, adapM, sobely,
          thr, 0, sobelyR]
titles = ['Canny', '', 'SobelX',
          'Laplacian', '','SobelY',
          'Global Thr (v=160)','' ,'Sobely Realce']
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
   
    plt.subplot(3,3,i*3+2),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
   
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows() 