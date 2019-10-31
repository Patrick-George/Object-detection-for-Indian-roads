import os
import cv2
import numpy as np 
import time
import random
import multiprocessing as mp
import shutil

#Size is image shape
#box = [box1_x1 box1_y1 box1_width box1_height]


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

def convert_to_normal(line,h,w,flip=False):
	
	x_center = line[0]* w
	y_center = line[1]* h
	box_width = line[2] * w
	box_height = line[3] * h

	x1 = int(x_center - (box_width/2.0))
	y1 = int(y_center - (box_height/2.0))

	x2 = int(x_center + (box_width/2.0))
	y2 = int(y_center + (box_height/2.0))

	coords = [x1,y1,x2,y2]

	if flip:
		#bottom left. New x2,y2
		x3 = abs(x1 - w)
		y3 = y2

		#Top right. New x1,y1. Y remains same
		x4 = abs(x2 - w)
		y4 = y1
		coords = [x4,y4,x3,y3]

		#coords = [x4,y4,x3,y3]
	return coords



def flip_image(img_file):		
	img = cv2.imread(img_file)
	horizontal_img = cv2.flip(img,1)
	
	label_file = img_file.replace(img_root,label_root).replace('jpg','txt').replace('png','txt')
	labels = open(label_file).readlines()

	boxes = []
	for x in labels:
		line = x.strip().split()
		#Remove class
		cls = line[0]
		line = line[1:]
		line = list(map(float,line))
		h,w,ch = img.shape
		#If flip is true,compute corords for lateral inversion		
		x1,y1,x2,y2 = convert_to_normal(line,h,w,flip=True)
		coords = [min(x1,x2),max(x1,x2),min(y1,y2),max(y1,y2)]

		#cv2.rectangle(horizontal_img,(x1,y1),(x2,y2),(0,255,0),3)
		relative_coords = convert_to_relative(img.shape,coords)
		if convert_to_normal(relative_coords,h,w) == [x1,y1,x2,y2]:
			print(img_file,' Done')

		boxes.append(cls + " "+ str(relative_coords[0]) + " " + str(relative_coords[1]) + " " + \
			str(relative_coords[2])+' '+str(relative_coords[-1])+'\n')

	with open(dest_label_root + label_file.split('/')[-1]+'_flip.txt'.replace(label_root,dest_label_root),'w') as out:
		for b in boxes:
			out.write(b)

	cv2.imwrite(dest_img_root + img_file.split('/')[-1]+'_flip.jpg'.replace(img_root,dest_img_root),horizontal_img)
	#shutil.copy(img_file,dest_img_root)
	#shutil.copy(label_file,dest_label_root)


def noisy(noise_type, image):
	if noise_type == "gauss":
	  row,col,ch= image.shape
	  mean = 0
	  var = np.random.uniform(0, 300)
	  sigma = var**0.5
	  gauss = np.random.normal(mean,sigma,(row,col,ch))
	  gauss = gauss.reshape(row,col,ch)
	  noisy = image + gauss
	  
	  return noisy
	
	elif noise_type == "s&p":
	  row,col,ch = image.shape
	  s_vs_p = 0.1
	  amount = 0.0005
	  out = np.copy(image)
	  
	  # Salt mode
	  num_salt = np.ceil(amount * image.size * s_vs_p)
	  coords = [np.random.randint(0, i - 1, int(num_salt))
			  for i in image.shape[0:2]]
	 
	  for i in range(len(coords[0])):
		  radius = int(np.random.uniform(1, 4))
		  cv2.circle(out, (coords[1][i], coords[0][i]), radius, tuple(np.random.uniform(230, 270, 3).astype('int')), -1)

	  # Pepper mode
	  num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
	  coords = [np.random.randint(0, i - 1, int(num_pepper))
			  for i in image.shape[0:2]]
	  
	  for i in range(len(coords[0])):
		  radius = int(np.random.uniform(1, 4))
		  cv2.circle(out, (coords[1][i], coords[0][i]), radius, tuple(np.random.uniform(0, 30, 3).astype('int')), -1)
	  
	  return out
  
	elif noise_type =="speckle":
		row,col,ch = image.shape
		gauss = np.random.randn(row,col,ch)/np.random.uniform(2, 10)
		gauss = gauss.reshape(row,col,ch)
		noisy = image + image * gauss
		return noisy
   
	elif noise_type == "motion_blur_horizontal":
		blur_size = 2*np.random.randint(2, 3) + 1
		kernel_motion_blur = np.zeros((blur_size, blur_size))
		kernel_motion_blur[int((blur_size-1)/2), :] = np.ones(blur_size)
		kernel_motion_blur = kernel_motion_blur / blur_size
		noisy = cv2.filter2D(image, -1, kernel_motion_blur)
		return noisy

	elif noise_type == "motion_blur_vertical":
		blur_size = 2*np.random.randint(2, 3) + 1
		kernel_motion_blur = np.zeros((blur_size, blur_size))
		kernel_motion_blur[:, int((blur_size-1)/2)] = np.ones(blur_size)
		kernel_motion_blur = kernel_motion_blur / blur_size
		noisy = cv2.filter2D(image, -1, kernel_motion_blur)
		return noisy

	elif noise_type == "resize":
		factor = 1.0 / 2
		scale = np.random.uniform(factor, 1.0)
		
		# Resize to the output scale
		resize_types = [cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]
		resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation = random.choice(resize_types))

		# Resize back tot he original scale
		noisy = cv2.resize(resized, (image.shape[1], image.shape[0]), interpolation = random.choice(resize_types))
		return noisy



def brighten(image):
	gamma = np.random.rand() + random.randint(0,2) + 0.35
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
	for i in np.arange(0, 256)]).astype("uint8")

	return cv2.LUT(image, table)

def darken(image):
	darken_param = np.random.uniform(1, 3.5)
	out = image / darken_param

	if np.random.binomial(1, 0.2):
		left_side = True
		if np.random.binomial(1, 0.5):
			left_side = False
		if (darken_param <= 3.5):
			p1_x = np.random.rand() * image.shape[1]
			p2_x = np.random.rand() * image.shape[1]
			shadow = np.random.uniform(1, 2.5)
			for x in range(image.shape[1]):
				for y in range(image.shape[0]):
					x_t = p1_x + ((p2_x - p1_x) / (image.shape[0] - 1)) * float(y)
					if x >= x_t and not left_side:
						out[y,x,:] /= shadow
					if x < x_t and left_side:
						out[y,x,:] /= shadow

	return out.astype(np.uint8)

def add_noise(img_file):

	noise_types = ['gauss','speckle', 'motion_blur_horizontal', 'motion_blur_vertical', 'resize', 'perspective']

	#for img_file in images:
	img = cv2.imread(img_file)

	print(img_file,' Noise Done')		
	
	label_file = img_file.replace(img_root,label_root).replace('jpg','txt').replace('png','txt')
	labels = open(label_file).readlines()

	noise_type_1 = random.choice(noise_types)
	remaining = [x for x in noise_types if noise_type_1!=x]
	noise_type_2 = random.choice(remaining)

	try:
		noisy_img_1 = noisy(noise_type_1,img)
		noisy_img_2 = noisy(noise_type_2,img)

		noisy_img_1 = noisy_img_1.astype(np.uint8)
		noisy_img_2 = noisy_img_2.astype(np.uint8)

		noisy_dark = darken(noisy_img_1)
		noisy_bright = brighten(noisy_img_2)

		with open(dest_label_root + label_file.split('/')[-1]+'_noisy_dark.txt'.replace(label_root,dest_label_root),'w') as out:
			for b in labels:
				out.write(b)				
		cv2.imwrite(dest_img_root + img_file.split('/')[-1]+'_noisy_dark.jpg'.replace(img_root,dest_img_root),noisy_dark)

		with open(dest_label_root + label_file.split('/')[-1]+'_noisy_bright.txt'.replace(label_root,dest_label_root),'w') as out:
			for b in labels:
				out.write(b)				
		cv2.imwrite(dest_img_root + img_file.split('/')[-1]+'_noisy_bright.jpg'.replace(img_root,dest_img_root),noisy_bright)

	except:
		pass

	return


#dest_img_root = 'train_images/'
#dest_label_root = 'train_labels/'

"""
folder_list = ['/home/traffic-model/training/19_7_18_split/extract_data/rbccps_prj01_d01/',\
				'/home/traffic-model/training/19_7_18_split/extract_data/rbccps_prj01_d02/',\
				'/home/traffic-model/training/19_7_18_split/extract_data/rbccps_prj01_d03/',\
				'/home/traffic-model/training/19_7_18_split/extract_data/rbccps_prj01_d04/'
				]
"""				
folder_list = ['/home/rbc-gpu/big_brother/yolo_data/TIKA_27818/extracted_data/mscoco_val/']

for f in folder_list:
	img_root = f 
	label_root = f

	dest_img_root = f
	dest_label_root = f

	images = os.listdir(img_root)
	images = list(map(lambda x: img_root+x,images))
	images = filter(lambda x:'jpg' in x,images)

	labels = os.listdir(label_root)
	labels = list(map(lambda z: label_root+z,labels))
	labels = filter(lambda x:'txt' in x,labels)

	p = mp.Pool(10)
	p.map(flip_image,images)

	#Flipping gives additional files
	images = os.listdir(img_root)
	images = list(map(lambda x: img_root+x,images))
	labels = os.listdir(label_root)
	labels = list(map(lambda z: label_root+z,labels))

	p = mp.Pool(10)
	p.map(add_noise,images)