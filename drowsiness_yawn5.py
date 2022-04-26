#python drowniness_yawn.py --webcam webcam_index
from datetime import datetime
from time import strftime
from distutils.command.config import config
from importlib.resources import path
from matplotlib.pyplot import text
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import pyrebase
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
from pygame import mixer
from sympy import false

mixer.init()
sound = mixer.Sound('sound/alarm.wav')
a = 0

# konfigurasi untuk menyambungkan ke firebase realtime DB
config = {
    "apiKey": "AIzaSyAdfa5RT8PrioqSOBNHB7FbSTcCZIkbhWM",
    "authDomain": "dms-tugasakhir.firebaseapp.com",
    "databaseURL":"https://dms-tugasakhir-default-rtdb.firebaseio.com/",
    "projectId": "dms-tugasakhir",
    "storageBucket": "dms-tugasakhir.appspot.com",
    "messagingSenderId": "851262891361",
    "appId": "1:851262891361:web:675128c61454872221d120"
}
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

# DB Realtime
db=firebase.database()
# my_image = "drowsiness.jpg"

#fungsi untuk menjalankan alarm menggunakan espeak
def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('call')
        s = 'espeak "'+msg+'"'
        os.system(s)

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

#fungsi untuk pembacaan ear (Eye Aspect Rasio)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

#
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # ear = (leftEAR + rightEAR) / 1.5      #lama
    ear = (leftEAR + rightEAR) / 2.0        #baru
    return (ear, leftEye, rightEye)

#jarak antara bibir atas dan bibir bawah
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 25
# 25, 30 dan 48
YAWN_THRESH = 20

alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

print("-> Menunnggu detector dan predictor")
#detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')    
#Faster but less accurate
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

print("-> Frame Sedang Diambil")
vs = VideoStream(src=args["webcam"]).start()

#vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi

time.sleep(1.0)

img_counter = 0
while True:
    a = a + 1

    frame = vs.read()
    frame = imutils.resize(frame, width=550)

    # contrast = 0.5
    # brightness = 6
    # gray[:,:,2] = np.clip(contrast * gray[:,:,2] + brightness, 0, 255)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # menampilkan date terkini
    date = cv2.putText(frame,str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),(10,400), font, 1,(0, 0, 255),2,cv2.LINE_AA)

    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

    #for rect in rects:
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        #Untuk membuat garis pada bagian mata
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Untuk membuat garis pada bibir
        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        nama = 'Riyadh'
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                sound.play()
                
                if alarm_status == False:
                    alarm_status = True
                    # t = Thread(target=alarm, args=('Wake Up Riyadh',))
                    # t.deamon = True
                    # t.start()

                    # img_name = 'drowsines/drowsiness_{}.jpg'.format(img_counter)
                    img_name ='capture-img/drowsines/Drowsiness_'+str(a)+'.jpg'
                    waktu = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))
                    img_counter += 1
                    cv2.waitKey(1000)
                    
                    #get url
                    
                    auth = firebase.auth()
                    email = "mhdriyadh367dms@gmail.com"
                    password = "qwerty123@@"
                    user = auth.sign_in_with_email_and_password(email, password)
                    
                    # url = storage.child(my_image).get_url(user['idToken'])
                    
                    subjek = ("Drowsiness")
                    cloudFile = 'Drowsiness/Drowsiness_'+str(a)+'.jpg'
                    file_local = str(img_name)
                    storage.child(cloudFile).put(file_local)
                    url = storage.child(cloudFile).get_url(None)

                    #push Data
                    data = {
                        
                        "namafile" : 'Drowsiness_'+str(a)+'.jpg',
                        "Subjek"  : (subjek),
                        "image" : (url),
                        "waktu" : str(waktu)
                    }
                    if ('data >= 0') :
                        db.child(nama).push(data)
                        print('sukses upload to cloud')
                        break

        else:
            COUNTER = 0
            alarm_status = False

        #Menguap
        if (distance > YAWN_THRESH):
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                sound.play()
                if alarm_status2 == False and saying == False:
                    alarm_status2 = True
                    # t = Thread(target=alarm, args=('take some fresh air sir',))
                    # t.deamon = True
                    # t.start()
                    
                    
                    # img_name = 'yawn/yawn_{}.png'.format(img_counter)
                    # cv2.imwrite(img_name, frame)
                    # print("{} written!".format(img_name))
                    # img_counter += 1
                    
                    img_name ="capture-img/yawn/Yawn_"+str(a)+".jpg"
                    waktu = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))
                    cv2.waitKey(1000)

                    auth = firebase.auth()
                    email = "mhdriyadh367dms@gmail.com"
                    password = "qwerty123@@"
                    user = auth.sign_in_with_email_and_password(email, password)
                    # url = storage.child(my_image).get_url(user['idToken'])
                    
                    subjek = ("Yawning")

                    # cloudFile = "Yawning/yawn_{}.png" .format(img_counter)
                    cloudFile = "Yawn/Yawn_"+str(a)+".jpg"
                    file_local = str(img_name)
                    storage.child(cloudFile).put(file_local)
                    url=storage.child(cloudFile).get_url(None)
                    
                    #push Data
                    data = {
                        "namafile" : "Yawn_"+str(a)+".jpg",
                        "Subjek"  : (subjek),
                        "image" : (url),
                        "waktu" : str(waktu)
                    }
                    if ('data >= 0') :
                        db.child(nama).push(data)
                        print('sukses upload to cloud')
                        break
                    
        else:
            alarm_status2 = False

        #Tulisan EAR dan Yawn pada Frame
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (400, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow('video frame', frame)
    # cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF


    # if alarm_status2|alarm_status == True :
    #     img_name = "opencv_frame_{}.png".format(img_counter)
    #     cv2.imwrite(img_name, frame)
    #     print("{} written!".format(img_name))
    #     img_counter += 1
    #     cv2.waitKey(2000)
    
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()