import cv2 as cv
from FaceDetection import FaceDetection
from FaceRecognition import FaceRecognitionThread

video_capture = cv.VideoCapture(0)
face_detection = FaceDetection()
face_recognition = FaceRecognitionThread(known_faces_dir="data/known_faces")


video_capture.start()
face_detection.start()
face_recognition.start()

cam_height = 750
cam_width = 500
cv.namedWindow("Face Detection App", cv.WINDOW_NORMAL)
cv.resizeWindow("Face Detection App", cam_width, cam_height)

while True:
    frame = video_capture.get_frame()
    if frame is None:
        break

    faces = face_detection.detect_faces(frame)

    for i in faces:
        x, y, width, height = i["box"]
        cv.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)


    frame = cv.flip(frame, 1)
    resized_frame = cv.resize(frame, (cam_width, cam_height))
    cv.imshow("Face Detection App", resized_frame)

    if cv.waitKey(1) == ord("q"):
        break

video_capture.release()
face_detection.stop()
face_recognition.stop()
cv.destroyAllWindows()
