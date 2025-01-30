import cv2 as cv
from mtcnn import MTCNN

cam = cv.VideoCapture(0)
detector = MTCNN()

cam_width = 720
cam_height = 500

cv.namedWindow("Face Recognition App", cv.WINDOW_NORMAL)
cv.resizeWindow("Face Recognition App", cam_width, cam_height)

while True:
    ret, frame = cam.read()

    output = detector.detect_faces(frame)

    for i in output:
        x, y, width, height = i["box"]
        cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1)

        left_eye = i["keypoints"]["left_eye"]
        right_eye = i["keypoints"]["right_eye"]
        nose = i["keypoints"]["nose"]
        mouth_left = i["keypoints"]["mouth_left"]
        mouth_right = i["keypoints"]["mouth_right"]

        cv.circle(frame, left_eye, 3, (0, 255, 0), -1)
        cv.circle(frame, right_eye, 3, (0, 255, 0), -1)
        cv.circle(frame, nose, 3, (0, 255, 0), -1)
        cv.circle(frame, mouth_left, 3, (0, 255, 0), -1)
        cv.circle(frame, mouth_right, 3, (0, 255, 0), -1)

    frame = cv.flip(frame, 1)

    resized_frame = cv.resize(frame, (cam_width, cam_height))
    cv.imshow("Face Recognition App", resized_frame)

    if cv.waitKey(1) == ord("q"):
        break

cam.release()
cv.destroyAllWindows()