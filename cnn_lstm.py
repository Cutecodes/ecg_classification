import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    """
    网络模型
    """
    def __init__(self, h=12,w=3000, num_classes=9):
        super(CNN_LSTM, self).__init__()
        # conv1: Conv2d -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=24, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2, stride=1),
        )
        # conv2: Conv2d -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=36, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm1d(36),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=48, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=48, out_channels=60, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=60, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=80, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.lstm1 = nn.Sequential(
            nn.LSTM(input_size = 91,hidden_size=60,num_layers = 2,batch_first = True,bidirectional=True),
        )
        # fully connected layer
        #self.dp1 = nn.Dropout(0.20)
        #self.fc1 = nn.Linear(95744, 25088)
        
        self.dp1 = nn.Dropout(0.30)
        self.fc1 = nn.Linear(16880, 4096)
        self.dp2 = nn.Dropout(0.30)
        self.fc2 = nn.Linear(4096, 2048)
        self.dp3 = nn.Dropout(0.30)
        self.fc3 = nn.Linear(2048,256)


        self.dp4 = nn.Dropout(0.30)
        self.fc4 = nn.Linear(256,num_classes)
        self.ReLU = nn.ReLU()
    
    def forward(self, x):
        """
        input: batchsize * 3 * image_size * image_size
        output: batchsize * num_classes
        """
        x = x.float()
        

        #print(x.shape)
        x = self.conv1(x)

        x = self.conv2(x)
        #print(x.shape)

        x = self.conv3(x)

        x = self.conv4(x)
       
        x = self.conv5(x)

        x = self.conv6(x)

        #print(x.shape)
        y = x
        x,(h_n, h_c)= self.lstm1(x)

        # view(x.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        #x = x.view(x.size(0), -1)
        x = x.reshape((x.shape[0],x.shape[1]*x.shape[2]))
        y = y.reshape((y.shape[0],y.shape[1]*y.shape[2]))
        x = torch.cat((x, y), 1)

        x = self.dp1(x)
        x = self.fc1(x)
        x = self.ReLU(x)
      
        x = self.dp2(x)
        x = self.fc2(x)
        x = self.ReLU(x)
       
        x = self.dp3(x)
        x = self.fc3(x)
        x = self.ReLU(x)

        x = self.dp4(x)
        x = self.fc4(x)
        #print(x.shape)

        output = self.ReLU(x)

        return output