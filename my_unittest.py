import cv2
import numpy as np

screen_width = 1440
image_width = 590



factor = 2.4
factor_y = 2.3
lines= [{'startX': 539.00397, 'startY': 141.2748, 'endX': 543.0243, 'endY': 288.9864, 'strokeWidth': 8}, {'startX': 543.0243, 'startY': 288.9864, 'endX': 538.59375, 'endY': 288.16797, 'strokeWidth': 8}]

img = cv2.imread('orignal.png')

scale_percent =100# percent of the original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dimension = (width, height)
print(f"Image width = {width} , height = {height}")
# resize image
img = cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)


print(img.shape)
mask = np.zeros(img.shape[:2], dtype="uint8")
print(mask.shape)
#cv2.line(mask, (10,10),(800,800) ,(255, 255, 255), 17)
for x in lines:
    print(f"StartX = {x['startX']*factor} , StartY = {x['startY']*factor_y} , EndX = {x['endX'] * factor}, EndY = {x['endY'] * factor_y}")
    cv2.line(mask, (int(x['startX'] * factor), int(x['startY'] * factor_y)),(int(x['endX'] * factor ), int(x['endY'] * factor_y)), (255, 255, 255), 32)
    cv2.line(img, (int(x['startX'] * factor), int(x['startY'] * factor_y )),(int(x['endX'] * factor ), int(x['endY'] * factor_y)), (255, 255, 255), 32)
    #cv2.line(mask, (10,10),(50,50) ,(255, 255, 255), 32)
    pass
    

masked = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("masked", masked)
cv2.imshow("mask", mask)
cv2.imshow("image", img)
cv2.waitKey(0)
