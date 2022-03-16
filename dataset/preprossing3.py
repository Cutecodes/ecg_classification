import pandas as pd 
import numpy as np
import os
import shutil
import scipy.io as sio


filelist = os.listdir(r'trainSet/') #列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
for file in filelist:
    try:
        data2 = sio.loadmat('trainSet/'+file)
    except:
        print(file)
