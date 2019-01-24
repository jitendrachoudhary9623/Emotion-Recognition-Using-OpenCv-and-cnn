import cv2
import numpy as np 
from FaceDetector import FaceDetector
video=cv2.VideoCapture(0)
faceDetector=FaceDetector()
while True:
    retvalue,frame=video.read()
    cv2.imshow("Video Face Detection",faceDetector.detectFaces(frame))
    if cv2.waitKey(1)==ord("q"):
        break

video.release()
cv2.destroyAllWindows()