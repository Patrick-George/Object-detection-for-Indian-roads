import os
import math
import shutil
import numpy as np
p = "0"
for fil in os.listdir("/media/patrick/Patrick/final_val_labels/"):
    flag = 1
    f = open("/media/patrick/Patrick/final_val_labels/"+fil,"r")
    #k = str(i)
    #f = open("/media/patrick/Patrick/modified_idd/test_images/"+k+".txt","r")
    fw = open("/media/patrick/Patrick/final_val_labels_large/"+fil,"w+")
    f1 = f.readlines()
    for x in f1:
        y = x.split(' ')
        bound_area = float(y[3])*float(y[4])
        if(bound_area > 0.001):
            fw.write(x)


# rootdir = "/home/patrick/IDD_Detection/labels"
# f = open("/home/patrick/IDD_Detection/label_test.txt","r")
# #w will finally have the images with small objects
# w = open("/home/patrick/IDD_Detection/small_objects.txt","w")
# f1 = f.readlines()
# for x in f1:
#     flag = 0
#     #z is just a temporary variable
#     z = x
#     x = x.strip()
#     if(os.path.exists(os.path.join(rootdir, x))):
#         f2 = open(os.path.join(rootdir, x),"r")
#         x2 = f2.readlines()
#         for x3 in x2:
#             y = x3.split(' ')
#             bound_area = float(y[3])*float(y[4])
#             if(bound_area < 0.00005):
#                 flag = 1
#                 break
#         if(flag == 1):
#             w.write(z)
