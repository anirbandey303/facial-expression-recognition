import cv2 #importing openCV
import sys #for system releated opperations
import numpy as np
from keras.preprocessing import image

from keras.models import model_from_json
model = model_from_json(open("../models/facial_expression_model_structure.json", "r").read())
model.load_weights('../models/facial_expression_model_weights.h5')

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') #xml for the eyes
video_capture = cv2.VideoCapture(0) #Inbuilt video camera for Video Stream



while True:
	retval, frame = video_capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=8,
		minSize=(35,35)
	)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,200), 2)
		detected_face = frame[int(y):int(y+h), int(x):int(x+w)]
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
		detected_face = cv2.resize(detected_face, (48,48))

		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)		 
		img_pixels /= 255		 
		predictions = model.predict(img_pixels)		 
		
		max_index = np.argmax(predictions[0]) 
		emotion = emotions[max_index]		 
		cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

		'''roi_gray = gray[int(y):int(y+h), int(x):int(x+w)] #locations of the face in grayscale
		roi_color = frame[int(y):int(y+h), int(x):int(x+w)] #locations of converted grayscale
		eyes = eye_cascade.detectMultiScale(roi_gray)# detect eyes
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0,255,0),2) # drawing a blue rectangle around the eyes'''

	cv2.imshow('Video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		video_capture.release()
		cv2.destroyAllWindows()
		sys.exit()