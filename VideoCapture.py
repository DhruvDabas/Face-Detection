import cv2 as cv
import threading

class VideoCaptureThread:
    def __init__(self, src=0):
        self.capture = cv.VideoCapture(src)
        self.capture.set(cv.CAP_PROP_BUFFERSIZE, 3)

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            else:
                print("Camera is closed")
                break

            if cv.waitKey(1) == ord("q"):
                break

        self.capture.release()
        cv.destroyAllWindows()

    def get_frame(self):
        return getattr(self, "frame", None)

    def release(self):
        self.capture.release()

video = VideoCaptureThread()

while True:
    frame = video.get_frame()
    if frame is not None:
        cv.imshow("Face Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
