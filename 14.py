import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread

def nothing(x):
    pass

cv2.namedWindow("Traking")
cv2.createTrackbar("P1", "Traking", 0, 255, nothing)
cv2.createTrackbar("P2", "Traking", 255, 255, nothing)
cv2.createTrackbar("P3", "Traking", 1, 20, nothing)


while True:
    img = cv2.imread("01.jpg", 0)

    p_1 = cv2.getTrackbarPos("P1", "Traking")
    p_2 = cv2.getTrackbarPos("P2", "Traking")
    p_3 = cv2.getTrackbarPos("P3", "Traking")

    im = cv2.imread("jpg//01.jpg", 1)

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    hsv = cv2.GaussianBlur(hsv, (7, 7), 0)

    th, bw  = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph   = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    dist    = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    borderSize = 75
    
    distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize, 
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    
    gap = 10                                
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
    kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap, 
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    
    distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
    
    mn, mx, _, _ = cv2.minMaxLoc(nxcor)
    
    th, peaks = cv2.threshold(nxcor, mx*0.5, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
    
    peaks8u = cv2.convertScaleAbs(peaks)
    
    contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    peaks8u = cv2.convertScaleAbs(peaks)    # to use as mask
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y+h, x:x+w], peaks8u[y:y+h, x:x+w])
        cv2.circle(im, (int(mxloc[0]+x), int(mxloc[1]+y)), int(mx), (255, 0, 0), 2)
        #cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 255), 2)
        #cv2.drawContours(im, contours, i, (0, 0, 255), 2)

    cv2.imshow('circles', im)
    #cv2.imshow('contours', contours)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
