import cv2 as cv
from mtcnn import MTCNN
import numpy as np

cam = cv.VideoCapture(0)
detector = MTCNN()

cam_width = 720
cam_height = 500

if not cam.isOpened():
    print("Error")
    exit()

cv.namedWindow("Face Recognition App", cv.WINDOW_NORMAL)
cv.resizeWindow("Face Recognition App", cam_width, cam_height)

while True:
    ret, frame = cam.read()

    output = detector.detect_faces(frame)

    for i in output:
        x, y, width, height = i["box"]

        center = (x + width // 2, y + height // 2)
        radius = max(width, height) // 2

        mask = np.zeros_like(frame)
        cv.circle(mask, center, radius, (255, 255, 255), -1)

        blurred_frame = cv.GaussianBlur(frame, (99, 99), 30)

        frame = np.where(mask == 255, blurred_frame, frame)

        # cv.circle(frame, center, radius, (0, 255, 0), 2)

    frame = cv.flip(frame, 1)
    resized_frame = cv.resize(frame, (cam_width, cam_height))
    cv.imshow("Face Recognition App", resized_frame)

    if cv.waitKey(1) == ord("q"):
        break

cam.release()
cv.destroyAllWindows()