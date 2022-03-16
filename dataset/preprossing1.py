import pandas as pd 
import numpy as np
import os
import shutil


data = pd.read_csv(r"label/REFERENCE.csv")
files1 = []
for i in range(len(data)):
	if(pd.notnull(data.at[i,'Second_label'])):
		files1.append(data.at[i,'Recording'])
print(len(files1))

for i in files1:
	shutil.move(r'trainSet/'+i+'.mat',r'other/'+i+'.mat')

