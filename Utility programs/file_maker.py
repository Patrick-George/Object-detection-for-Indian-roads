import os
import math
import shutil
import numpy as np

fw = open("/media/patrick/Patrick/modified_idd/test_without_small_jpg.txt","w+")
for i in range(1,1005):
    fw.write("/media/patrick/Patrick/modified_idd/test_images/test_without_small/"+str(i)+".jpg\n")
    