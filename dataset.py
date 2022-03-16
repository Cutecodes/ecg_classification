from torch.utils.data import Dataset,random_split
import torch
import scipy.io as sio
import pandas as pd
import os
import numpy as np
from numpy import random
from torchvision.transforms import transforms




class ECG(Dataset):
    def __init__(self,root,train=True,transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        if self.train == True:
            self.kind = "trainSet"
        else:
            self.kind = "testSet"

        self.labels_path = os.path.join(root,"label/")
        self.ecg_path = os.path.join(root,self.kind)

        self.filelist = os.listdir(self.ecg_path)
        df = pd.read_csv(self.labels_path+"REFERENCE.csv")
        self.labels = dict(zip(df['Recording'],df['First_label']))

    def __len__(self):
        if self.train == True:
            return len(3*self.filelist)
        return len(5*self.filelist)

    def __getitem__(self,index):
        file = self.filelist[index%len(self.filelist)]
        filepath = os.path.join(self.ecg_path,file)
        ecg = sio.loadmat(filepath)['ECG'][0][0][2]
    	#计算特征
        #hrv = ManFeat_HRV(ecg[1])
        #截取6s
        x = random.randint(len(ecg[0]))
        if x+3000>=len(ecg[0]):
            x=0
        #s = (len(ecg[0])-3000)//9
        #l = []
        #for i in range(10):
            #l.append(ecg[:,0*s:0*s+3000])
        #ecg = np.array(l)
        #ecg = ecg.reshape((12,30000))
        ecg = ecg[:,x:x+3000]
        ##ecg = ecg.astype(np.float32)
        target = self.labels[file[:5]]-1
        
        #ecg = transforms.ToTensor()(ecg)
        if self.transform is not None:
            ecg = self.transform(ecg)

        return ecg,target
        #return ecg,target,hrv.extract_features()

def main():
    import matplotlib.pyplot as plt
    #train_dataset = ECG(root='./dataset/',train=True)
    val_dataset = ECG(root='./dataset/',train=False)
    nums = [0 for i in range(9)]
    count = 0
    print(val_dataset[2])
    print(val_dataset[0][0].shape)
    for i in range(len(val_dataset)):
        nums[val_dataset[i][1]]+=1
        count+=1
    print(nums,count)
    val_dataset = ECG(root='./dataset/',train=True)
    nums = [0 for i in range(9)]
    count = 0
    print(val_dataset[2])
    print(val_dataset[0][0].shape)
    for i in range(len(val_dataset)):
        nums[val_dataset[i][1]]+=1
        count+=1
    print(nums,count)
    return 
    #datasets,_ = random_split(dataset=train,lengths=[3,7],generator=torch.Generator().manual_seed(0))
    #print(len(datasets))
    y,l = val_dataset[22]
    #print(l)
    x = [i for i in range(y.shape[1])]
    plt.subplot(621)
    plt.plot(x,y[0])
    plt.show()
if __name__ == '__main__':
	main()