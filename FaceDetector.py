from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import cv2
import numpy as np
import sys


    	
class FaceDetector():
  def __init__(self):
    self.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    self.detection_model_path = 'haarcascade_frontalface_default.xml'
    self.emotion_model_path = 'emotion_1.hdf5'
    self.EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
    self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
    #self.keypoints=load_model('model1.h5',compile=False) 

  def detectFaces(self,image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=self.detect(image)

    if len(faces) > 0:
      faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
      (fX, fY, fW, fH) = faces
      roi = gray[fY:fY + fH, fX:fX + fW]
      roi = cv2.resize(roi, (48, 48))
      roi = roi.astype("float") / 255.0
      roi = img_to_array(roi)
      roi = np.expand_dims(roi, axis=0)
      preds = self.emotion_classifier.predict(roi)[0]
      emotion_probability = np.max(preds)
      label = self.EMOTIONS[preds.argmax()]
      y_a=40
      output=zip(preds,self.EMOTIONS)
      for i in output:
        cv2.putText(image, " {} : {} ".format(i[1],i[0]*100), (15, y_a - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 2)
        y_a=y_a+20
      print(preds)
      cv2.putText(image, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 100, 0), 2)
      cv2.rectangle(image, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
    return image

		
  def detect(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)):
    boundaries = self.faceCascade.detectMultiScale(image,scaleFactor=scaleFactor,minNeighbors=minNeighbors,minSize=minSize,flags=cv2.CASCADE_SCALE_IMAGE)

    return boundaries

	
    		


		