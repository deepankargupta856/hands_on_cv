import cv2
import numpy as np

canvas = np.zeros((512,512,3),dtype=np.uint8)

cv2.rectangle(canvas,(100,100),(400,400),(255,0,0),-1)
cv2.putText(canvas,"annotations",(99,99),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
cv2.imshow("canvas",canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()