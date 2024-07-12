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
    if result:
        if result.boxes[0].cls ==0:
            print("There is person")
            print(result.boxes[0].conf)
            # print(result.boxes[0].xywh)

            bbox = result.boxes[0].xywh

            bbox = np.array(bbox)

            x, y, w, h = int((bbox[0][0])/2), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])

            print(x, y, w, h)

            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


            # print(x, y, w, h)

    cv.imshow("webcam", frame)
    # cv.imshow("webcam", frame)
    
    key = cv.waitKey(1)
    if key == ord("q"):
        break
    
cv.destroyAllWindows()