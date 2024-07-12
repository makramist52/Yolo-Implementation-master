from ultralytics import YOLO
import cv2 as cv
import numpy as np
import math

cap = cv.VideoCapture(0)

# 1. create the model, by giving weights, (nano, medium, large), just type the name, it will download the weights
# yolov8n.pt "yolo version 8 n means nano version"
model = YOLO("../YOLO-Weights/yolov8n.pt")

# Find Function
# x is the raw distance y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

coff = np.polyfit(x, y, 2)    # y = Ax^2 + BX + C

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

                # cv.circle(frame, (centerX+40, centerY+40), 10, (0, 255, 255), cv.FILLED)
                x1, x2, y1, y2 = centerX, centerX+40, centerY, centerY+40
                # Distance
                distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
                A, B, C = coff
                distanceCM = int(A*distance**2 + B*distance + C)
                
                cv.putText(frame, f"{distanceCM} CM", (centerX, centerY), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                i +=1

    cv.imshow("webcam", frame)
    # cv.imshow("webcam", frame)
    
    key = cv.waitKey(1)
    if key == ord("q"):
        break
    
cv.destroyAllWindows()