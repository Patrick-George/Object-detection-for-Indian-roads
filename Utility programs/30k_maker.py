import os
import shutil
import numpy as np
import matplotlib.pyplot as plt



rootdir = "/home/patrick/IDD_Detection/labels"
graph = [0 for i in range(0,12)]
#The flag is for keeping track of the classes that exceeded 30k images
flag = [0 for i in range(0,12)]
write_per = 1
write_per_ulti = 1
print(graph)

x_axis = ["person","animal","rider","motorcycle","bicycle","autorickshaw","car","truck","bus","vehicle fallback","traffic sign","traffic light"]

foo = open("/home/patrick/IDD_Detection/new_label_train.txt","w")
f = open("/home/patrick/IDD_Detection/label_train_shuf.txt","r")
f1 = f.readlines()
for x in f1:
    write_per = 1
    write_per_ulti = 0
    #z is just a temporary variable
    z = x
    x = x.strip()
    if(os.path.exists(os.path.join(rootdir, x))):
        f2 = open(os.path.join(rootdir, x),"r")
        x2 = f2.readlines()
        for x3 in x2:
            y = x3.split(' ')
            # print(int(x3[0]))
            graph[int(y[0])] += 1
            if(graph[int(y[0])] > 30000):
                flag[int(y[0])] = 1
            if(flag[int(y[0])] == 1):
                write_per = 0
            if(graph[int(y[0])] < 5000):
                write_per_ulti = 1
    if(write_per_ulti == 1):
        foo.write(z)
    elif(write_per == 1):
        foo.write(z)
            

print(graph)



# plt.bar(x_axis,graph,align='center') # A bar chart
# plt.xlabel('classes')
# plt.ylabel('Frequency')
# plt.show()
