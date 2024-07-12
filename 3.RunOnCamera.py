from ultralytics import YOLO
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

# 1. create the model, by giving weights, (nano, medium, large), just type the name, it will download the weights
# yolov8n.pt "yolo version 8 n means nano version"
model = YOLO("../YOLO-Weights/yolov8n.pt")

while True:
    
    Success, frame = cap.read()

    results = model.predict(frame)
    
    result = results[0]
    i = 1
    if result:
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            print("Object type:", class_id)
            print("Coordinates:", cords)
            print("Probability:", conf)
            print("---")

            if class_id == 'person' and conf > 0.70:
                cv.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), (0,255,0),2)
                cv.putText(frame, f"Person:{i}", (cords[0], cords[1]+20), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 2)
                cv.putText(frame, f"Probability:{conf*100}%", (cords[0], cords[1]+45), cv.FONT_HERSHEY_PLAIN, 1, (0,150,200), 2)
                
                # finding the center point of each person
                centerX = int((cords[0]+cords[2])/2)
                centerY = int((cords[1]+cords[3])/2)

                cv.circle(frame, (centerX, centerY), 10, (0, 255, 255), cv.FILLED)

                i +=1

    cv.imshow("webcam", frame)
    # cv.imshow("webcam", frame)
    
    key = cv.waitKey(1)
    if key == ord("q"):
        break
    
cv.destroyAllWindows()