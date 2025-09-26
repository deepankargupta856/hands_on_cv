import cv2 
from ultralytics import YOLO

cap = cv2.VideoCapture('video.mp4')

model = YOLO('yolov8n.pt')
while True:
    ret, frame = cap.read()
    results = model(frame,classes=[2,6,8])
    annotated_frame = results[0].plot()
    cv2.imshow("Annotated_video: ",annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()