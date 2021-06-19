from imutils.video import VideoStream
from imutils import face_utils
from gpiozero import Buzzer
import RPi.GPIO as GPIO
from time import sleep
import numpy as np
import imutils
import time
import dlib
import cv2



GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
buzzer = 23
GPIO.setup(buzzer, GPIO.OUT)

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 16
COUNTER = 0
ALARM_ON = False
alarm_buzzer = True
predictor_path = ''
cascade_path = ''




def euclidean_dist(point_a, point_b):
    """ compute and return the euclidean distance between the two
     points """
    return np.linalg.norm(point_a - point_b)


def eye_aspect_ratio(eye):
    """compute the euclidean distances between the two sets of
    vertical and the horizontal eye landmarks (x, y)-coordinates"""

    a = euclidean_dist(eye[1], eye[5])
    b = euclidean_dist(eye[2], eye[4])
    c = euclidean_dist(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear


def fps(frame, p_time):
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)


detector = cv2.CascadeClassifier(cascade_path)
predictor = dlib.shape_predictor(predictor_path)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

try:
  vs = VideoStream(usePiCamera=True).start()
  #vs = VideoStream(src=0).start()
except:
  print("Camera was not initialize")


p_time = 0

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in rects:

    	# construct a dlib rectangle object from the Haar cascade 
    	# bounding box
        rect = dlib.rectangle(int(x), int(y), int(x + w),
                              int(y + h))

        # determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
        shape = predictor(gray, rect)
        shape_array = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape_array[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
		# visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (225, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (225, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # if the eyes were closed for a sufficient number of
			# frames, then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True

                    # check to see if the  buzzer should
                    # be sounded
                    GPIO.output(buzzer, GPIO.HIGH)
                    print("Beep")
                    sleep(0.5)  # Delay in seconds
                    GPIO.output(buzzer, GPIO.LOW)
                    print("No Beep")
                    sleep(0.5)
                # draw an alarm on the frame
                cv2.putText(frame, "DROWSINESS ALERT !!!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False
            
        cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			    
    fps(frame=frame, p_time=p_time)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
