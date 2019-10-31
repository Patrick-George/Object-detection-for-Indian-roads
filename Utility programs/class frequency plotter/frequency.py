import os
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt


def entropy_list(x):
    total = sum(x)
    entropy = 0
    y = x
    for i in range(0,12):
        y[i] = y[i]/total
    

    for i in range(0,12):
        entropy += y[i]*math.log10(1/y[i])
    
    print(entropy)

rootdir = "/media/patrick/Patric/final_train_labels_large/"
graph = [0 for i in range(0,12)]
print(graph)

x_axis = ["person","animal","rider","motorcycle","bicycle","autorickshaw","car","truck","bus","traffic sign","traffic light"]

f = open("/media/patrick/Patrick/train_labels_final.txt","r")
f1 = f.readlines()
for x in f1:
    x = x.strip()
    if(os.path.exists(os.path.join(rootdir, x))):
        f2 = open(os.path.join(rootdir, x),"r")
        x2 = f2.readlines()
        for x3 in x2:
            y = x3.split(' ')
            # print(int(x3[0]))
            graph[int(y[0])] += 1

print(graph)
# print("The entropy is:")
# entropy_list(graph)
# entropy_list([1,1,1,1,1,1,1,1,1,1,1,10000000])


# plt.bar(x_axis,graph,align='center') # A bar chart
# plt.xlabel('classes')
# plt.ylabel('Frequency')
# plt.show()
