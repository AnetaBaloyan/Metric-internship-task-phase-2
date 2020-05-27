import cv2
import os
from camera_detector import detect


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

    length = len(os.listdir(path))
    current = 0
    failed = 0
    print('Classifying directory {} ...'.format(path))

    for file_name in os.listdir(path):

        # Draws the progress bar.
        current = current + 1
        done = int((current / length) * 100)
        loading = '[' + '#' * done + '.' * (100 - done) + ']' + str(done) + '%'
        print('\r', loading, end='')

        if file_name.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
            img = cv2.imread(path + '/' + file_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            try:
                canvas, isgood = detect(gray, img)
                if isgood:
                    cv2.imwrite(path + '/good/%s' % file_name, img)
                else:
                    cv2.imwrite(path + '/bad/%s' % file_name, img)
            except:
                failed = failed + 1
                pass

    print('')
    print('Finished classification. Failed: {} Succeeded: {}'.format(failed, length - failed))
    print('You can find the photos in {} and {}.'.format(path + '/good', path + '/bad'))
