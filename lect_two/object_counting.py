import cv2 
from ultralytics import YOLO
import numpy as np
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('conveyor.mp4')

unique_ids = set()
while True:
    
    ret, frame = cap.read()
    
    results = model.track(frame,classes=[41],persist=True,verbose=False)
    
    annotated_frame = results[0].plot()
    
    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.numpy()
        for oid in ids :
            unique_ids.add(oid)
        cv2.putText(annotated_frame,f"Count: {len(unique_ids)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
        cv2.imshow("Annotated_tracker: ",annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    