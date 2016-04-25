import cv2
import numpy as np
import numpy as np
from array import array
import argparse


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#recognizer = cv2.createEigenFaceRecognizer()
recognizer = cv2.createLBPHFaceRecognizer()
#recognizer = cv2.createFisherFaceRecognizer()

class trainData:
	def __init__(self,arg):
		if arg==0:
			self.load()
		else:
			self.loadimage()

	#Reads Faces from a Video
	def load(self):
		vidcap = cv2.VideoCapture('input.mp4')
		success,image = vidcap.read()
		train=[]
		label=[]
		while success:			
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)
			for (x,y,w,h) in faces:
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
				scaled_face=cv2.resize(gray[y:y+h, x:x+w], (100,100), interpolation = cv2.INTER_AREA)
				cv2.imwrite('face-'+'-'+str(x)+'.jpg', scaled_face)
				scaled_face = np.array(scaled_face)
			
			if cv2.waitKey(1) == 27:
				break

		recognizer.train(train, np.array(label))

	#Loads Images from the Database
	def loadimage(self):
		train=[]
		label=[]
		
		for i in range(1,7):
			for j in range(6):
				
				img = cv2.imread('in1/'+str(j)+str(i)+'.jpg')				
				
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				
				faces = face_cascade.detectMultiScale(gray, 1.3, 5)
				for (x,y,w,h) in faces:					
					scaled_face=cv2.resize(gray[y:y+h, x:x+w], (100,100), interpolation = cv2.INTER_AREA)
				
					train.append(scaled_face)
					label.append(j)

		recognizer.train(train, np.array(label))

#Does query on an input video	
class testData:
	def __init__(self,input_video):
		self.load(input_video)

	CURRENT_FRAME_FLAG = cv2.cv.CV_CAP_PROP_POS_FRAMES

	def load(self,input_video):
		CURRENT_FRAME_FLAG = cv2.cv.CV_CAP_PROP_POS_FRAMES
		vidcap = cv2.VideoCapture('query.mp4')
		success,image = vidcap.read()
		while success:
			cf = vidcap.get(CURRENT_FRAME_FLAG) - 1
			vidcap.set(CURRENT_FRAME_FLAG, cf+10)
			success,img = vidcap.read()	
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)

			for (x,y,w,h) in faces:
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
				scaled_face=cv2.resize(gray[y:y+h, x:x+w], (100,100), interpolation = cv2.INTER_AREA)
				predicted, conf = recognizer.predict(scaled_face)
				# conf=float(conf)
				print (conf)
				cv2.putText(img,str(predicted)+'-'+str(conf),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
				cv2.imshow('video', img)
			
			if cv2.waitKey(1000) == 27:
				break


a = trainData(2)
b = testData(8)
