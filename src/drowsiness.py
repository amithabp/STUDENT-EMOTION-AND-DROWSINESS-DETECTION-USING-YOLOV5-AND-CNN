import cv2
# from src.dbconn import *
import dlib
import pyttsx3
from scipy.spatial import distance
from _thread import start_new_thread
from tkinter import *
import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Combobox
import tkinter.ttk as ttk
import os

import time

import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import model_from_json

model = model_from_json(open("model/facial_expression_model_structure.json", "r").read())
model.load_weights('model/facial_expression_model_weights.h5')  # load weights
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

from scipy.ndimage import rotate

# INITIALIZING THE pyttsx3 SO THAT
# ALERT AUDIO MESSAGE CAN BE DELIVERED
engine = pyttsx3.init()

# SETTING UP OF CAMERA TO 1 YOU CAN
# EVEN CHOOSE 0 IN PLACE OF 1
cap = cv2.VideoCapture(0)

# FACE DETECTION OR MAPPING THE FACE TO
# GET THE Eye AND EYES DETECTED
face_detector = dlib.get_frontal_face_detector()

# PUT THE LOCATION OF .DAT FILE (FILE FOR
# PREDECTING THE LANDMARKS ON FACE )
dlib_facelandmark = dlib.shape_predictor(
	r"shape_predictor_68_face_landmarks.dat")

# FUNCTION CALCULATING THE ASPECT RATIO FOR
# THE Eye BY USING EUCLIDEAN DISTANCE FUNCTION
def Detect_Eye(eye):
	poi_A = distance.euclidean(eye[1], eye[5])
	poi_B = distance.euclidean(eye[2], eye[4])
	poi_C = distance.euclidean(eye[0], eye[3])
	aspect_ratio_Eye = (poi_A+poi_B)/(2*poi_C)
	return aspect_ratio_Eye


# MAIN LOOP IT WILL RUN ALL THE UNLESS AND
# UNTIL THE PROGRAM IS BEING KILLED BY THE USER
count=0
emocount=0
while True:
	null, frame = cap.read()
	gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_detector(gray_scale)

	for face in faces:
		face_landmarks = dlib_facelandmark(gray_scale, face)
		leftEye = []
		rightEye = []

		# THESE ARE THE POINTS ALLOCATION FOR THE
		# LEFT EYES IN .DAT FILE THAT ARE FROM 42 TO 47
		for n in range(42, 48):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			rightEye.append((x, y))
			next_point = n+1
			if n == 47:
				next_point = 42
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

		# THESE ARE THE POINTS ALLOCATION FOR THE
		# RIGHT EYES IN .DAT FILE THAT ARE FROM 36 TO 41
		for n in range(36, 42):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			leftEye.append((x, y))
			next_point = n+1
			if n == 41:
				next_point = 36
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

		# CALCULATING THE ASPECT RATIO FOR LEFT
		# AND RIGHT EYE
		# print(rightEye,"++___+++___+++")
		right_Eye = Detect_Eye(rightEye)
		left_Eye = Detect_Eye(leftEye)
		Eye_Rat = (left_Eye+right_Eye)/2

		# NOW ROUND OF THE VALUE OF AVERAGE MEAN
		# OF RIGHT AND LEFT EYES
		Eye_Rat = round(Eye_Rat, 2)

		# THIS VALUE OF 0.25 (YOU CAN EVEN CHANGE IT)
		# WILL DECIDE WHETHER THE PERSONS'S EYES ARE CLOSE OR NOT
		# print(Eye_Rat)
		if Eye_Rat < 0.25:
			count=count+1
			if count>=30:
				cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
							cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
				cv2.putText(frame, "Alert!!!! WAKE UP DUDE", (50, 450),
							cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)

				# CALLING THE AUDIO FUNCTION OF TEXT TO
				# AUDIO FOR ALERTING THE PERSON
				engine.say("Alert!!!! WAKE UP DUDE")
				engine.runAndWait()

		else:
			count=0
	img=frame
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	# print(faces) #locations of detected faces
	emotion = None

	for (x, y, w, h) in faces:


			cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw rectangle to main image

			detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
			detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis=0)

			img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

			predictions = model.predict(img_pixels)  # store probabilities of 7 expressions

			# find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
			max_index = np.argmax(predictions[0])

			emotion = emotions[max_index]

			print(emotion)


	cv2.imshow("RATING GENERATOR", frame)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
