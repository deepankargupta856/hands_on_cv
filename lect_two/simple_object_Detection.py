import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
image = cv2.imread("image.png")
results = model(image)
annotated_img = results[0].plot()
cv2.imshow("annotated image:",annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()