'''
绘制曲线、可视化
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from numpy import random

def showSample():
    filepath = r'./dataset/trainSet/A0011.mat'
    ecg = sio.loadmat(filepath)['ECG'][0][0]
    print("Sex:",ecg[0][0])
    print("Age:",ecg[1][0][0])
    curve = ecg[2]
    x = random.randint(len(curve[0]))
    if x+3000>=len(curve[0]):
        x=0
    curve = curve[:,x:x+3000]
    #x，y范围
    #plt.ylim((-2, 3))
    #plt.xlim(2,20)
    #x，y刻度
    #plt.xticks(np.arange(-2,3,0.5))
    #plt.xticks(np.arange(2,20,1))
    #x轴的标签
    t = np.array([i for i in range(curve.shape[1])])
    t = t/500
    plt.subplot(621)
    plt.plot(t,curve[0])
    plt.ylabel("I",size=10)

    plt.subplot(622)
    plt.plot(t,curve[1])
    plt.ylabel("II",size=10)
    plt.subplot(623)
    plt.plot(t,curve[2])
    plt.ylabel("III",size=10)
    plt.subplot(624)
    plt.plot(t,curve[3])
    plt.ylabel("aVR",size=10)
    plt.subplot(625)
    plt.plot(t,curve[4])
    plt.ylabel("aVL",size=10)
    plt.subplot(626)
    plt.plot(t,curve[5])
    plt.ylabel("aVF",size=10)
    plt.subplot(627)
    plt.plot(t,curve[6])
    plt.ylabel("VI",size=10)

    plt.subplot(628)
    plt.plot(t,curve[7])
    plt.ylabel("V2",size=10)
    plt.subplot(629)
    plt.plot(t,curve[8])
    plt.ylabel("V3",size=10)
    plt.subplot(6,2,10)
    plt.plot(t,curve[9])
    plt.ylabel("V4",size=10)
    plt.subplot(6,2,11)
    plt.plot(t,curve[10])
    plt.ylabel("V5",size=10)
    plt.xlabel("Time(second)",size=10)
    plt.subplot(6,2,12)
    plt.plot(t,curve[11])
    plt.ylabel("V6",size=10)
    plt.xlabel("Time(second)",size=10)
    
    #图例
    #plt.legend(loc='best')
    #plt.title("点个赞",size=22)

    plt.show()
    #print(ecg)

    

def showCurve():

    file = open('train_accuracy','r')
    train_acc = file.readlines()
    file.close()
    file = open('valid_accuracy','r')
    valid_acc = file.readlines()
    train_acc = [float(i) for i in train_acc]
    valid_acc = [float(i) for i in valid_acc]
    epoch = [i for i in range(len(train_acc))]
    
    plt.plot(epoch,train_acc,label='Train Accuracy',linewidth=2.5)
    plt.plot(epoch,valid_acc,label='Valid Accuracy')
    plt.ylabel("Accuracy",size=10)
    plt.xlabel("Epoch",size=10)
    plt.yticks(np.arange(0,100,10))
    plt.legend(loc='best')
    plt.show()

    file = open('train_loss','r')
    train_loss = file.readlines()
    file.close()
    file = open('valid_loss','r')
    valid_loss = file.readlines()
    train_loss = [float(i) for i in train_loss]
    valid_loss = [float(i) for i in valid_loss]
    epoch = [i for i in range(len(train_loss))]
    
    plt.plot(epoch,train_loss,label='Train Loss',linewidth=2.5)
    plt.plot(epoch,valid_loss,label='Valid Loss')
    plt.ylabel("Loss",size=10)
    plt.xlabel("Epoch",size=10)
    plt.yticks(np.arange(0,2,0.1))
    plt.legend(loc='best')
    plt.show()

def main():
    showSample()
    showCurve()
    #showTrainAcc()
    #showTestAcc()

if __name__ == '__main__':
    main()