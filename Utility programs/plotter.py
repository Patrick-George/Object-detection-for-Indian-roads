#!/usr/bin/env python
import sys
import cv2
from random import randint
import os


class plotter(object):
    def __init__(self, classes_path):
        with open(classes_path, 'r') as cp:
            self.classes = cp.readlines()
        self.images = []

    def plot_box(self, image_path, annotations_path):
        with open(annotations_path, 'r') as ap:
            annotations = ap.readlines()

        ano = [annot.strip("\n").split() for annot in annotations]

        

        for obj in ano:
            class_num = obj[0]
            obj[0] = self.classes[int(class_num)].strip('\n')

        self.image = cv2.imread(image_path, 3)
        H, W, channels = self.image.shape

        for obj in ano:
            x_box, y_box = int(float(obj[1])*W), int(float(obj[2])*H)
            x_wid, y_hei = int(float(obj[3])*W), int(float(obj[4])*H)

            top_left_x, top_left_y = int(x_box - x_wid/2), int(y_box + y_hei/2)
            bot_right_x, bot_right_y = int(x_box + x_wid/2), int(y_box - y_hei/2)

            cv2.rectangle(self.image, (top_left_x, top_left_y), (bot_right_x, bot_right_y), (randint(0, 255), randint(0, 255), randint(0, 255)), 3)
            #cv2.putText(self.image, obj[0], (x_box, y_box), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        self.images.append([image_path.strip().split('/')[-1], self.image])
        

    def plot_view(self):

        cv2.namedWindow("splot window")
        cv2.startWindowThread()
        cv2.imshow("splot window", self.image)
        cv2.waitKey(0)

        # if cv2.waitKey(50000) & 0xFF == ord('q'):
        #     print('pressed q')
        #     cv2.destroyAllWindows()

    def plot_save_all(self, rp):
        for iname, img in self.images:
            print("saving ", iname)
            cv2.imwrite(os.path.join(rp, iname), img)



if __name__ == '__main__':
    class_path = sys.argv[1]
    image_path = sys.argv[2]
    annotations_path = sys.argv[3]

    p = plotter(class_path)
    p.plot_box(image_path, annotations_path)
    p.plot_view()
