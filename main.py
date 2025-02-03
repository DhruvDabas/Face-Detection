import cv2

# all threads
video_capture = VideoCaptureThread(src=0)
face_detection = FaceDetectionThread()
face_recognition = FaceRecognitionThread(known_faces_dir="data/known_faces")


# Start threads
video_capture.start()
face_detection.start()
face_recognition.start()


while True:
    frame = video_capture.read()
    if frame is None:
        break

    recognized_faces = face_recognition.get_recognized_faces()

    for name, (x, y, w, h) in recognized_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Detection and Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.stop()
face_detection.stop()
face_recognition.stop()
cv2.destroyAllWindows()