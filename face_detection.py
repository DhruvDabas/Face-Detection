# Face detection

import cv2 as cv
import mtcnn as MTCNN

cam = cv.VideoCapture(0)
face_detector = MTCNN.MTCNN()

cam_width = 750
cam_height = 500

cv.namedWindow("Face Recognition ", cv.WINDOW_NORMAL)
cv.resizeWindow("Face Recognition ", cam_width, cam_height)

while True:
    ret, frame = cam.read()

    output = face_detector.detect_faces(frame)

    for i in output:
        x, y, width, height = i["box"]
        cv.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)


        # landmarks for face

        left_eye = i["keypoints"]["left_eye"]
        right_eye = i["keypoints"]["right_eye"]
        nose = i["keypoints"]["nose"]
        mouth_left = i["keypoints"]["mouth_left"]
        mouth_right = i["keypoints"]["mouth_right"]

        # cv.circle(frame, left_eye, 3, (0, 255, 0), -1)       (frame radius, color, fill)
        # cv.circle(frame, right_eye, 3, (0, 255, 0), -1)
        # cv.circle(frame, nose, 3, (0, 255, 0), -1)
        # cv.circle(frame, mouth_left, 3, (0, 255, 0), -1)
        # cv.circle(frame, mouth_right, 3, (0, 255, 0), -1)

        # lines to connect keypoints
        # cv.line(frame, left_eye, nose, (0, 0, 255), 2)
        # cv.line(frame, right_eye, nose, (0, 0, 255), 2)
        # cv.line(frame, nose, mouth_left, (0, 0, 255), 2)
        # cv.line(frame, nose, mouth_right, (0, 0, 255), 2)


    frame = cv.flip(frame, 1) # flip the frame horizontally 

    resized_frame = cv.resize(frame, (cam_width, cam_height))
    cv.imshow("Face Recognition App", resized_frame)

    if cv.waitKey(1) == ord("q"):
        break

cam.release()
cv.destroyAllWindows()