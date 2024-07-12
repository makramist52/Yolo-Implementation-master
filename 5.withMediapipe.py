from ultralytics import YOLO
import cv2 as cv
import numpy as np
import serial
# from cvzone import SerialModule

cap = cv.VideoCapture(0)

# 1. create the model, by giving weights, (nano, medium, large), just type the name, it will download the weights
# yolov8n.pt "yolo version 8 n means nano version"
model = YOLO("../YOLO-Weights/yolov8n.pt")

selected_person = None 
# Define the mapping for servo motor angles based on your physical setup
# This is just an example, you need to adjust it based on your system
x_angle_map = np.interp(np.arange(0, 640), [0, 640], [0, 180])
y_angle_map = np.interp(np.arange(0, 480), [0, 480], [0, 180])
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
            # print("Object type:", class_id)
            # print("Coordinates:", cords)
            # print("Probability:", conf)
            # print("---")

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
    # Allow the user to select a person by entering a number
    key = cv.waitKey(1) & 0xFF
    if key != 255:  # Check if a key is pressed
        key -= ord('0')  # Convert ASCII to integer
        if 1 <= key <= i:  # Check if the entered number is valid
            selected_person = key
            print(f"Selected Person: {selected_person}")

    # Display the selected person continuously
    if selected_person is not None:
        selected_person_cords = result.boxes[selected_person - 1].xyxy[0].tolist()
        selected_person_cords = [round(x) for x in selected_person_cords]
        x, y, w, h = selected_person_cords
        person_crop = frame[y:h, x:w]
        cv.imshow("Selected Person", person_crop)

        # Convert centerX and centerY to angles for servo motors
        x_angle = int(np.interp(centerX, [0, 640], [0, 180]))
        y_angle = int(np.interp(centerY, [0, 480], [0, 180]))

        print(f"x angel: {x_angle}, y angle: {y_angle}")

    key = cv.waitKey(1)
    if key == ord("q"):
        break
    
cv.destroyAllWindows()