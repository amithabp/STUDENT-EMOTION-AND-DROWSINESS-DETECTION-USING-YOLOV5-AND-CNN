
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




