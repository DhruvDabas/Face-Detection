import cv2 as cv

cam = cv.VideoCapture(0)

cam_width = 720
cam_height = 500

if not cam.isOpened():
    print("Error")
    exit()

cv.namedWindow("Face Recognition App", cv.WINDOW_NORMAL)
cv.resizeWindow("Face Recognition App", cam_width, cam_height)

while True:
    ret, frame = cam.read()

    if not ret:
        print("Error")
        break

    frame = cv.flip(frame, 1)

    resized_frame = cv.resize(frame, (cam_width, cam_height))

    cv.imshow("Face Recognition App", resized_frame)

    if cv.waitKey(1) == ord("q"):
        break

cam.release()
cv.destroyAllWindows()