from mtcnn import MTCNN
import cv2 as cv

detector = MTCNN()

img = cv.imread("/path")

output = detector.detect_faces(img)
print(output)

for i in output:
    x, y, width, height = i["box"]
    left_eyeX, left_eyeY = i["keypoints"]["left_eye"]
    right_eyeX, right_eyeY = i["keypoints"]["right_eye"]
    noseX, noseY = i["keypoints"]["nose"]
    mouth_leftX, mouth_leftY = i["keypoints"]["mouth_left"]
    mouth_rightX, mouth_rightY = i["keypoints"]["mouth_right"]


    cv.circle(img, (left_eyeX, left_eyeY), 3, (0, 255, 0), -1)
    cv.circle(img, (right_eyeX, right_eyeY), 3, (0, 255, 0), -1)
    cv.circle(img, (noseX, noseY), 3, (0, 255, 0), -1)
    cv.circle(img, (mouth_leftX, mouth_leftY), 3, (0, 255, 0), -1)
    cv.circle(img, (mouth_rightX, mouth_rightY), 3, (0, 255, 0), -1)

    cv.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 1)

cv.imshow("window", img)
cv.waitKey(0)