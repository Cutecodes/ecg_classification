import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from dataset import ECG
from torch.utils.data import DataLoader,Dataset,TensorDataset,random_split,ConcatDataset
from torchvision.transforms import transforms
import torch
import torch.nn as nn
#from model import CNN
import os
from cnn_lstm import CNN_LSTM
from model3 import CNN
import warnings 
warnings.filterwarnings('ignore')

def plot_matrix(cm, labels_name, title="Confusion Matrix", thresh=0.8, axis_labels=None):

# 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

# 图像标题
    if title is not None:
        plt.title(title)
# 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):            
            plt.text(j, i, format(cm[i][j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
# 显示
    plt.show()

'''
Score the prediction answers by comparing answers.csv and REFERENCE.csv in validation_set folder,
The scoring uses a F1 measure, which is an average of the nice F1 values from each classification
type. The specific score rules will be found on http://www.icbeb.org/Challenge.html.
Matrix A follows the format as:
                                     Predicted
                      Normal  AF  I-AVB  LBBB  RBBB  PAC  PVC  STD  STE
               Normal  N11   N12   N13   N14   N15   N16  N17  N18  N19
               AF      N21   N22   N23   N24   N25   N26  N27  N28  N29
               I-AVB   N31   N32   N33   N34   N35   N36  N37  N38  N39
               LBBB    N41   N42   N43   N44   N45   N46  N47  N48  N49
Reference      RBBB    N51   N52   N53   N54   N55   N56  N57  N58  N59
               PAC     N61   N62   N63   N64   N65   N66  N67  N68  N69
               PVC     N71   N72   N73   N74   N75   N76  N77  N78  N79
               STD     N81   N82   N83   N84   N85   N86  N87  N88  N89
               STE     N91   N92   N93   N94   N95   N96  N97  N98  N99

For each of the nine types, F1 is defined as:
Normal: F11=2*N11/(N1x+Nx1) AF: F12=2*N22/(N2x+Nx2) I-AVB: F13=2*N33/(N3x+Nx3) LBBB: F14=2*N44/(N4x+Nx4) RBBB: F15=2*N55/(N5x+Nx5)
PAC: F16=2*N66/(N6x+Nx6)    PVC: F17=2*N77/(N7x+Nx7)    STD: F18=2*N88/(N8x+Nx8)    STE: F19=2*N99/(N9x+Nx9)

The final challenge score is defined as:
F1 = (F11+F12+F13+F14+F15+F16+F17+F18+F19)/9

In addition, we alse calculate the F1 measures for each of the four sub-abnormal types:
            AF: Faf=2*N22/(N2x+Nx2)                         Block: Fblock=2*(N33+N44+N55)/(N3x+Nx3+N4x+Nx4+N5x+Nx5)
Premature contraction: Fpc=2*(N66+N77)/(N6x+Nx6+N7x+Nx7)    ST-segment change: Fst=2*(N88+N99)/(N8x+Nx8+N9x+Nx9)

The static of predicted answers and the final score are saved to score.txt in local path.
'''

def score(model, val_loader):

    A = np.zeros((9, 9), dtype=np.float)
    model.eval()
    correct = 0
    with torch.no_grad():
        l = len(val_loader)//5
        for i, (ecgs, targets) in enumerate(val_loader): 
            if i>=l:
                break         
            #optimizer.zero_grad()
            outputs = model(ecgs)
            for j in range(1,5):
                outputs+= model(val_loader[j*l+i][0])
            maxnums,preds = torch.max(outputs,1)
            # sum up batch loss 
            #for idx in range(len(preds)):
                #A[targets[idx]][preds[idx]] += 1
            A[targets][preds]+=1
            correct += (preds == targets).sum().item()

    print(A)
    plot_matrix(A,labels_name=[0,1,2,3,4,5,6,7,8],axis_labels=["Normal","AF","I-AVB","LBBB","RBBB","PAC","PVC","STD","STE"])
    F11 = 2 * A[0][0] / (np.sum(A[0, :]) + np.sum(A[:, 0]))
    F12 = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
    F13 = 2 * A[2][2] / (np.sum(A[2, :]) + np.sum(A[:, 2]))
    F14 = 2 * A[3][3] / (np.sum(A[3, :]) + np.sum(A[:, 3]))
    F15 = 2 * A[4][4] / (np.sum(A[4, :]) + np.sum(A[:, 4]))
    F16 = 2 * A[5][5] / (np.sum(A[5, :]) + np.sum(A[:, 5]))
    F17 = 2 * A[6][6] / (np.sum(A[6, :]) + np.sum(A[:, 6]))
    F18 = 2 * A[7][7] / (np.sum(A[7, :]) + np.sum(A[:, 7]))
    F19 = 2 * A[8][8] / (np.sum(A[8, :]) + np.sum(A[:, 8]))

    F1 = (F11+F12+F13+F14+F15+F16+F17+F18+F19) / 9

    ## following is calculating scores for 4 types: AF, Block, Premature contraction, ST-segment change.

    Faf = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
    Fblock = 2 * (A[2][2] + A[3][3] + A[4][4]) / (np.sum(A[2:5, :]) + np.sum(A[:, 2:5]))
    Fpc = 2 * (A[5][5] + A[6][6]) / (np.sum(A[5:7, :]) + np.sum(A[:, 5:7]))
    Fst = 2 * (A[7][7] + A[8][8]) / (np.sum(A[7:9, :]) + np.sum(A[:, 7:9]))

    # print(A)
    print('Accuracy:%.2f%%'%(correct*100/np.sum(A)))
    print('Total File Number: ', np.sum(A))

    print("F11: ", F11)
    print("F12: ", F12)
    print("F13: ", F13)
    print("F14: ", F14)
    print("F15: ", F15)
    print("F16: ", F16)
    print("F17: ", F17)
    print("F18: ", F18)
    print("F19: ", F19)
    print("F1: ", F1)

    print("Faf: ", Faf)
    print("Fblock: ", Fblock)
    print("Fpc: ", Fpc)
    print("Fst: ", Fst)

    with open('score.txt', 'w') as score_file:
        # print (A, file=score_file)
        print('Accuracy:%.2f%%'%(correct*100/np.sum(A)),file=score_file)
        print ('Total File Number: %d\n' %(np.sum(A)), file=score_file)
        print ('F11: %0.3f' %F11, file=score_file)
        print ('F12: %0.3f' %F12, file=score_file)
        print ('F13: %0.3f' %F13, file=score_file)
        print ('F14: %0.3f' %F14, file=score_file)
        print ('F15: %0.3f' %F15, file=score_file)
        print ('F16: %0.3f' %F16, file=score_file)
        print ('F17: %0.3f' %F17, file=score_file)
        print ('F18: %0.3f' %F18, file=score_file)
        print ('F19: %0.3f\n' %F19, file=score_file)
        print ('F1: %0.3f\n' %F1, file=score_file)
        print ('Faf: %0.3f' %Faf, file=score_file)
        print ('Fblock: %0.3f' %Fblock, file=score_file)
        print ('Fpc: %0.3f' %Fpc, file=score_file)
        print ('Fst: %0.3f' %Fst, file=score_file)

        score_file.close()

if __name__ == '__main__':
    net = CNN_LSTM()
    if os.path.exists('./model.pth'):
        try:
            net.load_state_dict(torch.load('./model.pth'))
        except Exception as e:
            print(e)
            print("Parameters Error")
    
    #train_dataset = ECG(root='./dataset/',train=True,transform=transforms.ToTensor())
    test_dataset = ECG(root='./dataset/',train=False,transform=transforms.ToTensor())
    #test_loader = DataLoader(test_dataset, 1, shuffle = False)
    score(net,test_dataset)


