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
import matplotlib.pyplot as plt
from collections import Counter
import cv2
# from src.dbconn import *
global stu
stu=0
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
from scipy.ndimage import rotate
from src.dbconnection import *
root=Tk()
root.geometry('780x550+20+0')
import pymysql
# -----------------------------
# face expression recognizer initialization
from tensorflow.keras.models import model_from_json
model = model_from_json(open("model/facial_expression_model_structure.json", "r").read())
model.load_weights('model/facial_expression_model_weights.h5')  # load weights
# -----------------------------
con=pymysql.connect(host='localhost',port=3306,user='root',password='',db='student_attention_system')
cmd=con.cursor()
# lb = tk.Listbox(root)
# lb.pack()
flag=False
rec_emotions=[]
def ff():
    stud = MAPPING[box.get()]
    global stu
    stu = stud
    start_new_thread( detect_emotion(),())

def st():
    root.destroy()
# for file in os.listdir():
#     if file.endswith(".mp4"):
#         lb.insert(0, file)

def detect_emotion():
    global stu
    emotioncount=0
    drdrcount=0
    framecount=0
    model = model_from_json(open("model/facial_expression_model_structure.json", "r").read())
    model.load_weights('model/facial_expression_model_weights.h5')  # load weights
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    # model = load_model('model.h5')
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
        aspect_ratio_Eye = (poi_A + poi_B) / (2 * poi_C)
        return aspect_ratio_Eye
    # MAIN LOOP IT WILL RUN ALL THE UNLESS AND
    # UNTIL THE PROGRAM IS BEING KILLED BY THE USER
    count = 0
    emocount = 0
    rec_emotions=[]
    emotions_detected = []
    while True:
        null, frame = cap.read()
        framecount+=1
        print(framecount,emotioncount)
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
                next_point = n + 1
                if n == 47:
                    next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                # cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

            # THESE ARE THE POINTS ALLOCATION FOR THE
            # RIGHT EYES IN .DAT FILE THAT ARE FROM 36 TO 41
            for n in range(36, 42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x, y))
                next_point = n + 1
                if n == 41:
                    next_point = 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                # cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

            # CALCULATING THE ASPECT RATIO FOR LEFT
            # AND RIGHT EYE
            # print(rightEye,"++___+++___+++")
            right_Eye = Detect_Eye(rightEye)
            left_Eye = Detect_Eye(leftEye)
            Eye_Rat = (left_Eye + right_Eye) / 2

            # NOW ROUND OF THE VALUE OF AVERAGE MEAN
            # OF RIGHT AND LEFT EYES
            Eye_Rat = round(Eye_Rat, 2)

            # THIS VALUE OF 0.25 (YOU CAN EVEN CHANGE IT)
            # WILL DECIDE WHETHER THE PERSONS'S EYES ARE CLOSE OR NOT
            # print(Eye_Rat)
            if Eye_Rat < 0.25:
                count = count + 1
                if count >= 30:
                    drdrcount+=1
                    # cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
                    #             cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
                    # cv2.putText(frame, "Alert!!!! WAKE UP DUDE", (50, 450),
                    #             cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
                    #
                    # # CALLING THE AUDIO FUNCTION OF TEXT TO
                    # # AUDIO FOR ALERTING THE PERSON
                    # engine.say("Alert!!!! WAKE UP DUDE")
                    # engine.runAndWait()


            else:

                count = 0
        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # print(faces) #locations of detected faces
        emotion = None
        if(len(faces))>0:
            emotioncount=emotioncount+1
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
            rec_emotions.append(emotion)
            emotions_detected.append(emotion)

        if emotioncount==50:
            print(rec_emotions)
            pos_cnt = rec_emotions.count("happy") + rec_emotions.count("neutral")+rec_emotions.count("surprise")
            ratio = pos_cnt / len(rec_emotions)
            print(ratio,"++++++++++++++++++++++")
            print(ratio,"++++++++++++++++++++++")
            print(ratio,"++++++++++++++++++++++")
            print(ratio,"++++++++++++++++++++++")

            q="insert into `rating` values (null,%s,%s,now())"
            val=(stu,ratio)
            iud(q,val)
            emotioncount=0
            rec_emotions=[]
        if framecount==150:
            avg=(drdrcount/150)*100
            print(avg,"++++++++++++++++")
            framecount=0
            drdrcount=0
            rat=5
            if avg>50:
                rat=0
            elif avg>40:
                rat=1
            elif avg>30:
                rat=2
            elif avg>20:
                rat=3
            elif avg>10:
                rat=4




            q="insert into `rating` values (null,%s,%s,now())"
            val=(stu,rat)
            iud(q,val)
            print("==========++++++++++")
            print("==========++++++++++")
            print("==========++++++++++")
            print("==========++++++++++")

        cv2.imshow("RATING GENERATOR", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()

    counter = Counter(emotions_detected)
    emotion_names = list(counter.keys())
    emotion_counts = list(counter.values())
    plt.bar(emotion_names, emotion_counts)
    plt.bar(emotion_names, emotion_counts, color='red')
    plt.title(f"Emotional Rating")
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.show()

    cv2.destroyAllWindows()


l1= Label(root,text="SELECT SUBJECT")
l1.place(relx=0.20,rely=0.25)
cmd.execute("SELECT `sub_id`,`subject` FROM `subject`")
s = cmd.fetchall()

# combo1 = Combobox(root,value=s[1])
# combo1.place(relx=0.35,rely=0.25)
# combo1.current(1)

box_value = StringVar()
# mydict = {'A': 'Haha What', 'B': 'Lala Sorry', 'C': 'Ohoh OMG'}
cmd.execute("SELECT `subject` FROM `subject`")
s = cmd.fetchall()
sname=[]
for i in s:
    sname.append(i[0])
mydict=sname
print("student-----",mydict)
cmd.execute("SELECT `sub_id` FROM `subject`")
s2 = cmd.fetchall()
id=[]
for i in s2:
    id.append(i[0])
print("key-------",id)
MAPPING= dict(zip(mydict,id))
print("mappind------",MAPPING)
# print("haii-",hai)
# MAPPING = {'Item 1' : 1, 'Item 2' : 2, 'Item 3' : 3}
# box_values=list(mydict.values())
box = ttk.Combobox(root, textvariable=box_value, values=mydict, state='readonly')
box.current(0)
box.grid(column=9, row=0,padx=270,pady=140)

# bstart = ttk.Button(root, text="START")
b2=Button(root,text="START",command=ff)
b2.place(relx=0.35,rely=0.65)
import threading



# bstart.pack()
# lb.insert(0,"chrome.mp4")
# bstart.bind("<ButtonPress-1>", ff)
b3=Button(root,text="STOP",command=st)
b3.place(relx=0.45,rely=0.65)
root.mainloop()
#DD

# https://www.youtube.com/watch?v=tFNJGim3FXw
import cv2
# from src.dbconn import *
import dlib
import pyttsx3
from scipy.spatial import distance

import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

engine = pyttsx3.init()

face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor(
	r"shape_predictor_68_face_landmarks.dat")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def direction(img):
    image = img

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            # print(y)
            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                flag = 1

                text = "Looking Down"
            else:

                text = "Forward"
            return text
    else:
        return  ""


def Detect_Eye(eye):
	poi_A = distance.euclidean(eye[1], eye[5])
	poi_B = distance.euclidean(eye[2], eye[4])
	poi_C = distance.euclidean(eye[0], eye[3])
	aspect_ratio_Eye = (poi_A+poi_B)/(2*poi_C)
	return aspect_ratio_Eye
count=0
cap = cv2.VideoCapture(0)
while cap.isOpened():
    x = []
    y = []
    x1 = []
    y1 = []
    conf = []
    nm = []

    ret, frame = cap.read()

    # Make detections
    results = model(frame)

    xmin=results.pandas().xyxy[0].get('xmin')
    ymin=results.pandas().xyxy[0].get('ymin')
    xmax=results.pandas().xyxy[0].get('xmax')
    ymax=results.pandas().xyxy[0].get('ymax')
    confidence=results.pandas().xyxy[0].get('confidence')
    name=results.pandas().xyxy[0].get('name')

    for i in range(len(xmin)):
        if name.get(i)=='person':
            x.append(xmin.get(i))
            y.append(ymin.get(i))
            x1.append(xmax.get(i))
            y1.append(ymax.get(i))
            nm.append(name.get(i))
    img=frame
    for i in range(len(x)):
        cv2.rectangle(frame, (int(x[i]), int(y[i])), (int(x1[i]), int(y1[i])), (255, 0, 0), 2)  # draw rectangle to main image

        detected_face = img[int(y[i]):int(y1[i]), int(x[i]):int(x1[i])]  # crop detected face
        # print(detected_face)

        directiontxt=direction(detected_face)
        cv2.imwrite("sample.png",detected_face)
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

        faces = face_detector(detected_face)

        for face in faces:
            face_landmarks = dlib_facelandmark(detected_face, face)
            leftEye = []
            rightEye = []

            # THESE ARE THE POINTS ALLOCATION FOR THE
            # LEFT EYES IN .DAT FILE THAT ARE FROM 42 TO 47
            for n in range(42, 48):
                xx = face_landmarks.part(n).x
                yy = face_landmarks.part(n).y
                rightEye.append((xx, yy))

                next_point = n + 1
                if n == 47:
                    next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                # cv2.line(frame, (xx, yy), (x2, y2), (0, 255, 0), 1)
            print(rightEye)
            # THESE ARE THE POINTS ALLOCATION FOR THE
            # RIGHT EYES IN .DAT FILE THAT ARE FROM 36 TO 41
            for n in range(36, 42):
                xx = face_landmarks.part(n).x
                yy = face_landmarks.part(n).y
                leftEye.append((xx, yy))
                next_point = n + 1
                if n == 41:
                    next_point = 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                # cv2.line(frame, (xx, yy), (x2, y2), (255, 255, 0), 1)
            print(leftEye)
            # CALCULATING THE ASPECT RATIO FOR LEFT
            # AND RIGHT EYE
            right_Eye = Detect_Eye(rightEye)
            left_Eye = Detect_Eye(leftEye)
            Eye_Rat = (left_Eye + right_Eye) / 2

            # NOW ROUND OF THE VALUE OF AVERAGE MEAN
            # OF RIGHT AND LEFT EYES
            Eye_Rat = round(Eye_Rat, 2)

            # THIS VALUE OF 0.25 (YOU CAN EVEN CHANGE IT)
            # WILL DECIDE WHETHER THE PERSONS'S EYES ARE CLOSE OR NOT
            print(Eye_Rat,"+++++++++++++++++++++")
            if Eye_Rat < 0.25:
                count = count + 1
                if count >= 4:
                    directiontxt="Drowsiness Detected "+directiontxt
                    # cv2.putText(frame, "Drowsiness Detected "+directiontxt, (int(x[i]), int(y[i]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # else:
                #     cv2.putText(frame,  directiontxt, (int(x[i]), int(y[i]) - 5),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)



            else:
                count = 0
        cv2.putText(frame, directiontxt, (int(x[i]), int(y[i]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)



    cv2.imshow('YOLO', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()





