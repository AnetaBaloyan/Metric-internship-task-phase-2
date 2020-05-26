import cv2
import os
from camera_detector import detect, clean_cascades, download_and_build_cascades


# Clears old results and creates 'good' and 'bad' folders.
def prep_dir(path):
    if os.path.exists(path + '/good'):
        # Remove previously detected good shots if any.
        for img in os.listdir(path + '/good'):
            if img.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
                os.remove(path + '/good/' + img)
    else:
        os.makedirs(path + '/good')

    if os.path.exists(path + '/bad'):
        # Remove previously detected bad shots if any.
        for img in os.listdir(path + '/bad'):
            if img.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
                os.remove(path + '/bad/' + img)
    else:
        os.makedirs(path + '/bad')


# Separates photos into 'good' and 'bad' folders based on whether those are
# worth a shot or not. You can see the detection squares in the outcome.
def classify(path):
    print('Preparing the directory...')
    prep_dir(path)
    clean_cascades()
    left_eye_cascade, right_eye_cascade, \
    face_cascade, smile_cascade = download_and_build_cascades()

    length = len(os.listdir(path))
    current = 0
    print('Classifying directory %s ...' % path)

    for file_name in os.listdir(path):

        # Draws the progress bar.
        current = current + 1
        done = int((current / length) * 100)
        loading = '[' + '#' * done + '.' * (100 - done) + ']' + str(done) + '%'
        print('\r', loading, end='')

        if file_name.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
            img = cv2.imread(path + '/' + file_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            canvas, isgood = detect(gray, img, left_eye_cascade,
                                    right_eye_cascade, face_cascade, smile_cascade)

            if isgood:
                cv2.imwrite(path + '/good/%s' % file_name, img)
            else:
                cv2.imwrite(path + '/bad/%s' % file_name, img)
    print('')
    print('Finished classification. You can find the photos in %s and %s.'
          % (path + '/good', path + '/bad'))
