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

    if not ret:
        print("Error")
        break

    output = detector.detect_faces(frame)

    for i in output:
        x, y, width, height = i["box"]

        # Calculate the center and radius of the circle
        center = (x + width // 2, y + height // 2)
        radius = max(width, height) // 2

        # Extract the face region
        face = frame[y:y+height, x:x+width]

        # Apply Gaussian blur to the face region
        blurred_face = cv.GaussianBlur(face, (99, 99), 30)

        # Replace the face region with the blurred face
        frame[y:y+height, x:x+width] = blurred_face

        # Draw a circle around the face
        cv.circle(frame, center, radius, (0, 255, 0), 2)  # Green circle with thickness 2

    # Flip the frame horizontally
    frame = cv.flip(frame, 1)

    # Resize the frame to the desired dimensions
    resized_frame = cv.resize(frame, (cam_width, cam_height))

    # Display the frame
    cv.imshow("Face Recognition App", resized_frame)

    # Exit on 'q' key press
    if cv.waitKey(1) == ord("q"):
        break

cam.release()
cv.destroyAllWindows()