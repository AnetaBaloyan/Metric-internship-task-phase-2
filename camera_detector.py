import cv2
import time
import os
import numpy as np
from scipy.spatial import distance as dist
import dlib
from imutils import face_utils
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import random

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
emotion_classifier = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
EMOTIONS = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']
COLORS = [(255, 0, 0)]


def clean_photos():
    # Remove previous photos.
    for pic in os.listdir('./captured_photos'):
        if pic.endswith('.jpg'):
            os.remove('./captured_photos/' + pic)


def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates.
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio.
    eye_ar = (A + B) / (2.0 * C)

    # Return the eye aspect ratio.
    return eye_ar


def random_colors(n):
    colors = [(0, 0, 0)] * n
    for i in range(n):
        rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors[i] = rgb
    return colors


# Detection function. Detects faces, eyes, smiles and returns the frame with
# a flag that tells if the frame is good to save or not.
def detect(gray, frame):
    EYE_AR_THRESH = 0.23

    # Grab the indexes of the facial landmarks for the left and
    # right eye, respectively.
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    face_rects = detector(gray, 0)
    shot_flag = len(face_rects) > 0  # The flag is True if at least one face is detected.

    # Make sure we have a distinct color fo reach face.
    global COLORS
    if len(face_rects) > len(COLORS):
        COLORS = random_colors(len(face_rects))

    # Detect eyes and smiles for each face.
    for i, face in zip(range(len(face_rects)), face_rects):
        f_col = COLORS[i]
        (x, y) = (face.tl_corner().x, face.tl_corner().y)
        w = face.width()
        h = face.height()
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), f_col, 2)
        roi_gray = gray[y:y + h, x:x + w]

        # Determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array.
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes.
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the aspect ratios of both eyes.
        l_eye_ar = eye_aspect_ratio(leftEye)
        r_eye_ar = eye_aspect_ratio(rightEye)

        both_eyes_open = True

        # Draw the left eye.
        if l_eye_ar < EYE_AR_THRESH:
            both_eyes_open = False
            col = (100, 100, 0)
        else:
            col = (255, 255, 0)

        leftEyeHull = cv2.convexHull(leftEye)
        cv2.drawContours(frame, [leftEyeHull], -1, col, 1)

        # Draw the right eye.
        if r_eye_ar < EYE_AR_THRESH:
            both_eyes_open = False
            col = (0, 100, 100)
        else:
            col = (0, 255, 255)

        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [rightEyeHull], -1, col, 1)

        try:
            roi_gray = cv2.resize(roi_gray, (64, 64))
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)

            prediction = emotion_classifier.predict(roi_gray)[0]
            label = EMOTIONS[prediction.argmax()]
            # Display emotion on the video.
            cv2.putText(frame, "Emotion: {}".format(label), (10, 120 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, f_col, 2)
            issmiling = label == 'happy'
        except:
            issmiling = False
            pass

        shot_flag = shot_flag and issmiling and both_eyes_open

    return frame, shot_flag


# Cleans the environment, starts the camera and the detector.
def start_detector():
    clean_photos()

    video_capture = cv2.VideoCapture(0)
    shot_flag = False
    end_time = -1
    while True:
        # Captures video_capture frame by frame
        _, frame = video_capture.read()

        # To capture image in monochrome
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not shot_flag:
            # calls the detect() function
            clean_frame = frame.copy()
            canvas, shot_flag = detect(gray, frame)

            if shot_flag:
                end_time = time.time() + 1
                name = str(time.time())
                cv2.imwrite('./captured_photos/%s.jpg' % name, frame)
                cv2.imwrite('./captured_photos/%s.jpg' % (name + '_clean'), clean_frame)
        else:
            if end_time != -1 and time.time() >= end_time:
                shot_flag = False
                end_time = -1

        # Displays the result on camera feed
        cv2.imshow('Video', frame)

        # The control breaks once q key is pressed
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # Release the capture once all the processing is done.
    video_capture.release()
    cv2.destroyAllWindows()
