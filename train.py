from dataset import ECG
from torch.utils.data import DataLoader,Dataset,TensorDataset,random_split,ConcatDataset,WeightedRandomSampler
from torchvision.transforms import transforms
import torch
import torch.nn as nn
#from model import CNN
import os
from cnn_lstm import CNN_LSTM

import warnings 
warnings.filterwarnings('ignore')

def get_kfold_data(k, i, X):

    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = len(X) // k  # 每份的个数:数据总条数/折数（组数）
    lengths = [fold_size]*(k-1)
    lengths.append(len(X)-fold_size*(k-1))
    subdataset = random_split(X,lengths = lengths)

    
    X_valid = subdataset[i]
    subdataset.pop(i)
    X_train = ConcatDataset(subdataset)
        
    return X_train, X_valid

def traink(model,optimizer,lossfunc,X_train, X_val, BATCH_SIZE,TOTAL_EPOCHS=1):
    
    train_loader = DataLoader(X_train, BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(X_val, BATCH_SIZE, shuffle = True)
    losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        train_loss = 0
        correct = 0   # 记录正确的个数，每个epoch训练完成之后打印accuracy
        for i,(ecgs, targets) in enumerate(train_loader):
            optimizer.zero_grad()        # 清零
            outputs = model(ecgs)
            maxnums,preds = torch.max(outputs,1)
            loss = lossfunc(outputs,targets)
            train_loss += loss.item() * len(targets)
            loss.backward()
            optimizer.step()
            # 计算正确率
            correct += (preds == targets).sum().item()
            if (i+1) % 5 == 0:
            # 每10个batches打印一次loss
                print ('Iter :%.4f%%,  Loss: %.4f'%((i + 1)*100/(len(X_train)/BATCH_SIZE), loss.item()))
        losses.append(train_loss/len(X_train))    
        accuracy = 100.*correct/len(X_train)
        print('Training set: Average Loss: {:.4f},accuracy: {}/{} ({:.3f}%)'.format(
            train_loss/len(X_train), correct, len(X_train), accuracy))
        train_acc.append(accuracy)
        
        
        # 每个epoch计算测试集accuracy
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for i, (ecgs, targets) in enumerate(val_loader):          
                optimizer.zero_grad()
                outputs = model(ecgs)
                maxnums,preds = torch.max(outputs,1)
               
                loss = lossfunc(outputs,targets)      # batch average loss
                val_loss += loss.item() * len(targets)             # sum up batch loss 
                correct += (preds == targets).sum().item()
        
        val_losses.append(val_loss/len(X_val))
        accuracy = 100.*correct/len(X_val)
        print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            val_loss/len(X_val), correct, len(X_val), accuracy))
        val_acc.append(accuracy)
        
        
    return losses, val_losses, train_acc, val_acc


def k_fold(k,model,optimizer,lossfunc,X,learning_rate = 0.0001,batch_size = 16):

    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum , valid_acc_sum = 0, 0
    train_loss_f = open("train_loss","a")
    train_accuracy_f = open("train_accuracy","a")
    valid_loss_f = open("valid_loss","a")
    valid_accuracy_f = open("valid_accuracy","a")

    for i in range(k):
        print('*'*25,'第', i + 1,'折','*'*25)
        data = get_kfold_data(k, i, X)    # 获取k折交叉验证的训练和验证数据

        train_loss, val_loss, train_acc, val_acc = traink(model, optimizer, lossfunc, data[0],data[1],batch_size) 
        for i in train_loss:
            train_loss_f.write("%s\n"%(i))
        for i in val_loss:
            valid_loss_f.write("%s\n"%(i))
        for i in train_acc:
            train_accuracy_f.write("%s\n"%(i))
        for i in val_acc:
            valid_accuracy_f.write("%s\n"%(i))
        
        print('train_loss:{:.5f}, train_acc:{:.3f}%'.format(train_loss[-1], train_acc[-1]))
        print('valid loss:{:.5f}, valid_acc:{:.3f}%\n'.format(val_loss[-1], val_acc[-1]))
        
        train_loss_sum += train_loss[-1]
        valid_loss_sum += val_loss[-1]
        train_acc_sum += train_acc[-1]
        valid_acc_sum += val_acc[-1]
    train_loss_f.close()
    train_accuracy_f.close()
    valid_loss_f.close()   
    valid_accuracy_f.close()
    print('\n', '#'*10,'k折交叉验证结果','#'*10) 
    

    print('average train loss:{:.4f}, average train accuracy:{:.3f}%'.format(train_loss_sum/k, train_acc_sum/k))
    print('average valid loss:{:.4f}, average valid accuracy:{:.3f}%'.format(valid_loss_sum/k, valid_acc_sum/k))

    return 




def main():
    lossfunc = nn.CrossEntropyLoss()
    lr =0.0001
    batch_size = 100
    num_classes = 9
    epoch = 50 
    k = 10
    net = CNN_LSTM()
    if os.path.exists('./model.pth'):
        try:
            net.load_state_dict(torch.load('./model.pth'))
        except Exception as e:
            print(e)
            print("Parameters Error")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    #train_dataset = ECG(root='./dataset/',train=True,transform=transforms.ToTensor())
    #class_sample_counts = [652, 696, 475, 129, 1084, 382, 437, 524, 129]
    train_dataset = ECG(root='./dataset/',train=True)
    #for i in range(len(train_dataset)):
        #class_sample_counts[train_dataset[i][1]]+=1

    
    
    for i in range(epoch):
        print("Epoch:{}".format(i))
        k_fold(k,net,optimizer,lossfunc,train_dataset,learning_rate = lr,batch_size = batch_size)
        torch.save(net.state_dict(),'./model.pth')



if __name__ == '__main__':
    main()