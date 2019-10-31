import os
import cv2
import numpy as np 
import time
import random
import multiprocessing as mp
import shutil


def convert_to_relative(size, box):

    dw = 1./size[1]
    dh = 1./size[0]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = abs(x*dw)
    w = abs(w*dw)
    y = abs(y*dh)
    h = abs(h*dh)
    return (x,y,w,h)

def convert_to_normal(line,h,w):
    x_center = line[0]* w
    y_center = line[1]* h
    box_width = line[2] * w
    box_height = line[3] * h

    x1 = int(x_center - (box_width/2.0))
    y1 = int(y_center - (box_height/2.0))

    x2 = int(x_center + (box_width/2.0))
    y2 = int(y_center + (box_height/2.0))

    coords = [x1,y1,x2,y2]
    return coords


def read_image(img_file):
    names = ["person", "animal", "rider", "motorcycle", "bicycle", "autorickshaw", "car", "truck", "bus", "truck", "traffic sign", "traffic light"]

    img = cv2.imread(img_file)
    label_file=img_file.replace(img_root,label_root).replace('jpg','txt').replace('png','txt')
    labels = open(label_file).readlines()

    boxes = []
    ht_wt = []		 #for storing the height and weight 
    filename = []  #for storing the name of the file
    path = [] #path of the file
    
    for x in labels:
        line = x.strip().split()
        #Remove class
        cls = line[0]
        line = line[1:]
        line = list(map(float,line))
        h,w,ch = img.shape
        
        x1,y1,x2,y2 = convert_to_normal(line,h,w)
        coords = [min(x1,x2),max(x1,x2),min(y1,y2),max(y1,y2)]
        #print(cls, coords)
        
        relative_coords = convert_to_relative(img.shape,coords)
        if convert_to_normal(relative_coords,h,w) == [x1,y1,x2,y2]:
            print(img_file,' Done')

        boxes.append("  <object>\n" + "   <name>" + names[int(cls)] + "</name>\n" + "   <bndbox>\n"+ "    <xmin>" + str(coords[0]) + "</xmin>\n" + "    <xmax>" + str(coords[1]) + "</xmax>\n" + "    <ymin>" + str(coords[2])+"</ymin>\n" + "    <ymax>" +str(coords[-1])+"</ymax>\n" + "   </bndbox>\n" + "  </object>\n")
        ht_wt.append("  <size>\n" + "    <width>" + str(w) + "</width>\n" + "    <height>" + str(h) + "</height>\n" + "  </size>")
        path.append("  <path>" + str(img_file) + "</path>\n")
        filename.append("  <filename>" + str(img_file.split('/')[-1]) + "</filename>\n")
        

        with open(dest_label_root + label_file.split('/') [-1].split('.')[0]+'.xml'.replace(label_root,dest_label_root),'w') as out:
            out.write("<annotation>\n")
            out.write("  <foder>MELON</folder>\n")
            out.write(filename[0])
            out.write(path[0])
            out.write(ht_wt[0])
            for b in boxes:
                out.write(b)
                
            out.write("</annotation>\n")
         
    
folder_list = ['/home/chandana/SSD-on-Custom-Dataset-ssd/data/VOCdevkit/MELON/validation/']

for f in folder_list:

    img_root = f 
    label_root = f
    

    dest_img_root = f
    dest_label_root = f

    images = os.listdir(img_root)
    images = list(map(lambda x: img_root+x,images))
    images = list(filter(lambda x:'jpg' in x,images))


    labels = os.listdir(label_root)
    labels = list(map(lambda z: label_root+z,labels))
    labels = list(filter(lambda x:'txt' in x,labels))

for i in range(len(images)):
    #print(images[i].split('/')[-1].split('.')[0])
    read_image(images[i])
    
    
print(i)
