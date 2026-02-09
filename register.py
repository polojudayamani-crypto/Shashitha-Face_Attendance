import cv2
import os

name = input("Enter Name: ")
path = f"dataset/{name}"

if not os.path.exists(path):
    os.makedirs(path)

cam = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Register Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f"{path}/{count}.jpg", gray)
        count += 1
        print("Image saved")

    if count == 20:
        break

cam.release()
cv2.destroyAllWindows()
