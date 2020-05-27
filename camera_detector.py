import cv2
import wget
import time
import os


def clean_photos():
    # Remove previous photos.
    for pic in os.listdir('./captured_photos'):
        if pic.endswith('.jpg'):
            os.remove('./captured_photos/' + pic)


def clean_cascades():
    # Remove previous detection xmls.
    for xml in os.listdir('.'):
        if xml.endswith('.xml'):
            os.remove(xml)


def download_and_build_cascades():
    # URLs of haarcascade xmls.
    face_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    smile_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml'
    left_eye_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_lefteye_2splits.xml'
    right_eye_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_righteye_2splits.xml'

    # Download the detectors.
    face = wget.download(face_url)
    smile = wget.download(smile_url)
    left_eye = wget.download(left_eye_url)
    right_eye = wget.download(right_eye_url)

    # Create the classifiers.
    left_eye_cascade = cv2.CascadeClassifier(left_eye)
    right_eye_cascade = cv2.CascadeClassifier(right_eye)
    face_cascade = cv2.CascadeClassifier(face)
    smile_cascade = cv2.CascadeClassifier(smile)

    return left_eye_cascade, right_eye_cascade, face_cascade, smile_cascade


# Detection function. Detects faces, eyes, smiles and returns the frame with
# a flag that tells if the frame is good to save or not.
def detect(gray, frame, left_eye_cascade, right_eye_cascade, face_cascade, smile_cascade):
    faces = face_cascade.detectMultiScale(gray, 1.25, 10)  # 1.3, 10
    shot_flag = len(faces) > 0  # The flag is True if at least one face is detected.

    # Detect eyes and smiles for each face.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 40)

        left_eyes = left_eye_cascade.detectMultiScale(roi_gray, 1.4, 20)
        right_eyes = right_eye_cascade.detectMultiScale(roi_gray, 1.4, 20)

        shot_flag = shot_flag and len(smiles) > 0 and len(left_eyes) > 0 \
                    and len(right_eyes) > 0

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)

        for (sx, sy, sw, sh) in left_eyes:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (255, 255, 0), 2)

        for (sx, sy, sw, sh) in right_eyes:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 255, 255), 2)

    return frame, shot_flag


# Cleans the environment, starts the camera and the detector.
def start_detector():
    clean_photos()
    clean_cascades()
    left_eye_cascade, right_eye_cascade, \
    face_cascade, smile_cascade = download_and_build_cascades()

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
            canvas, shot_flag = detect(gray, frame, left_eye_cascade,
                                       right_eye_cascade, face_cascade, smile_cascade)

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
