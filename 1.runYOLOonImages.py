from ultralytics import YOLO
import cv2 as cv

i = 1
while True:
    img = f"images/{i}.jpg"

    image = cv.imread(img)

    # 1. create the model, by giving weights, (nano, medium, large), just type the name, it will download the weights
    # yolov8n.pt "yolo version 8 n means nano version"
    model = YOLO("../YOLO-Weights/yolov8n.pt")

    results = model(img, show=True)
    cv.imshow("Original Image", image)

    key = cv.waitKey(0)
    if key == ord("n"):
        i +=1
        if i == 7:
            i = 1
    else:
        break
    
cv.destroyAllWindows()