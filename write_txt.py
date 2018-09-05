# -*-coding:utf-8 -*-
import os
pic_dir="/home/liuyp/liu/keras_mobilenet/data3/train"
textfilename = "/home/liuyp/liu/keras_mobilenet/data3/train_txt.txt"
if not os.path.exists(textfilename):
    f1=open(textfilename, 'w')
    f1.close()
f1 = open(textfilename, "r+")
for root1,dir1,file1 in os.walk(pic_dir):
  
    print("###################")
 
    for f in file1:
        root_name=root1.split('/')
        path = os.path.join(root_name[-1], f)
        #print("aa is",path)
        f1.read()
        f1.write('/'+path +' '+root_name[-1] + "\n")
f1.close()
