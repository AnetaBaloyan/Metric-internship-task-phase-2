from dataset_detector import classify, prep_dir
from camera_detector import start_detector, clean_photos


#############################################################
# To run the detector on a sample dataset, un-comment the   #
# following code.                                           #
#############################################################
# path = './test_dataset'  # The path to the dataset.
# classify(path)


#############################################################
# To run the detector with a live camera, un-comment the    #
# following code.                                           #
#############################################################
start_detector()
