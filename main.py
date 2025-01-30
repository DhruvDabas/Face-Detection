from mtcnn import MTCNN
import cv2 as cv

detector = MTCNN()

img = cv.imread("/Users/demohatesrng/Desktop/TensorFlow/Face_Detection/rathee.jpg")

output = detector.detect_faces(img)
print(output)

x, y, width, height = output[0]["box"]

cv.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

cv.imshow("window", img)
cv.waitKey(0)

