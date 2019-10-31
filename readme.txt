OBJECT DETECTION FOR AN AUTONOMOUS VEHICLE FOR INDIAN ROADS

COLLABORATORS: PATRICK GEORGE(Email: 111601015@smail.iitpkd.ac.in, pattypatrician@gmail.com), 
               CHANDANA S.K(111701009@smail.iitpkd.ac.in,chandanashreekrishna@gmail.com)


This folder consists of a dataset for Indian roads.
The classes in the dataset are namely:

person
animal
rider
motorcycle
bicycle
autorickshaw
car
truck
bus
traffic sign
traffic light


This dataset is the modified version of the IDD dataset published by IIIT Hyderabad, which had the classes:

bicycle
bus
traffic sign
train
motorcycle
car
traffic light 
person
vehicle fallback
truck 
autorickshaw
animal
caravan
rider
trailer

Some of the classes were removed to improve the detection accuracy of the remaining, more significant classes.
Since the class imbalance of the original training set was very high, a program was made to cut down the number of images that had objects which had many instances.
Since the test set that IDD dataset came with had no annotations, we cut 2000 images from the validation set for testing.
Some images of animal and bicycle classes have been removed as they were not good enough to improve accuracy.
The lables which were in xml format was converted to darknet format txt files.


This folder includes the modified dataset images and label files for testing, training and validation.

The utility programs include the following codes:

frequency.py:  To calculate the number of instances of each class occuring in a dataset. 

XmlToTxt: To convert the xml annotation files to text files of darknet format.

30k_maker.py: Since the class imbalance was high among the classes, this program was made to go through the images and pick a set of images such that the class imbalance is reduced and the entropy of the classes was near 1. So, running this program on a dataset produces a new dataset with better equality between the number of instances of the classes.

image_gen.py: Does some augmentations to the images

object_finder.py: Lists the images with objects whose bounding boxes are smaller than a specified threshold.

rotate.py: Rotates the images to a certain angle to produce new images for training

small_remover: Given the labels of a dataset, removes the bounding boxes which are smaller than a specified threshold.

plotter.py: Given the image and the label file, prints an image with the bounding boxes plotted.

txt_to_xml.py: converts the txt files of annotations in darknet format to xml files.

move.py: Given a text file with paths to some files, the program moves all the files to folder after renaming those files.
