import os
import shutil

rootdir = "/home/chandana/third_set/JPEGImages/"
copypath = "/home/chandana/test_images/test_folder_ordinary/"


f = open("/home/chandana/third_set/test_new.txt","r")
f1 = f.readlines()
count = 1
for x in f1:
    x = x.strip()
    shutil.copy(os.path.join(rootdir,x),"/home/chandana/test_images/intermediate/")
    for filename in os.listdir("/home/chandana/test_images/intermediate/"):
        os.rename(os.path.join("/home/chandana/test_images/intermediate/",filename),os.path.join("/home/chandana/test_images/intermediate/",(str(count)+".txt")))
    for filename in os.listdir("/home/chandana/test_images/intermediate/"):
        shutil.move(os.path.join("/home/chandana/test_images/intermediate/",filename),"/home/chandana/test_images/test_folder_ordinary/")
    count += 1    
    
    
try:
    shutil.copy(source, target)
except IOError as e:
    print("Unable to copy file. %s" % e)
except:
    print("Unexpected error:", sys.exc_info())
