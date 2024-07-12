from ultralytics import YOLO
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
model = YOLO("../YOLO-Weights/yolov8n.pt")

people_dict = {}  # Dictionary to store unique IDs for people
person_id_counter = 1  # Counter for assigning unique IDs

while True:
    success, frame = cap.read()

    results = model.predict(frame)
    result = results[0]

    if result:
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)

            if class_id == 'person' and conf > 0.70:
                cv.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), (0, 255, 0), 2)
                cv.putText(frame, f"Person:{person_id_counter}", (cords[0], cords[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 0, 255), 2)
                cv.putText(frame, f"Probability:{conf * 100}%", (cords[0], cords[1] + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 150, 200), 2)

                centerX = int((cords[0] + cords[2]) / 2)
                centerY = int((cords[1] + cords[3]) / 2)
                cv.circle(frame, (centerX, centerY), 10, (0, 255, 255), cv.FILLED)

                # Assign a unique ID to each person based on their bounding box coordinates
                person_id = people_dict.get((cords[0], cords[1], cords[2], cords[3]))
                if person_id is None:
                    person_id = person_id_counter
                    people_dict[(cords[0], cords[1], cords[2], cords[3])] = person_id
                    person_id_counter += 1

                print(f"Person:{person_id} - Coordinates:{cords}")

    cv.imshow("webcam", frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break

cv.destroyAllWindows()
