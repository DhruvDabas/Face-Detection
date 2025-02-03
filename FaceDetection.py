import mtcnn as MTCNN
import cv2 as cv

class FaceDetectionThread:
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self,frame):
        ret, frame = self.read()
        output = self.detect_faces(frame)

        for i in output:
            x, y, width, height = i["box"]
            cv.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
        
        return self.detector.detect_faces(frame)