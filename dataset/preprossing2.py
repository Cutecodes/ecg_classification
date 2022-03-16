import pandas as pd 
import numpy as np
import os
import shutil

mcnt = 0

filelist = os.listdir(r'trainSet/') #列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
for file in filelist:
    if np.random.random_sample()<0.3:
        src = os.path.join(r'trainSet/', file)
        dst = os.path.join(r'testSet/', file) 
        shutil.move(src, dst)
        mcnt += 1
print(len(filelist))
print("%f %%moved"%(mcnt*100/len(filelist)))

