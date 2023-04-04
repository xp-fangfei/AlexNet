import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary




class AlexNet(nn.Module):             #继承nn.Module这个父类
    def __init__(self, num_classes=2, init_weights=False):
        super(AlexNet, self).__init__()
        # 用nn.Sequential()将网络打包成一个模块
        self.features = nn.Sequential(
            # Conv2d(in_channels, out_channels, kernel_size, stride, padding, ...)
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2) , # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),  #直接修改覆盖原值
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  #展平后再传入全连接层

        x = self.classifier(x)

        return x
    #网络权重初始化，实际上pytorch再构建网络时会自动初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): #判断是否是卷积层
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')  #用何凯明法初始化权重

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)        #初始化偏重为0
            elif isinstance(m, nn.Linear):          #若是全连接层
                nn.init.normal_(m.weight, 0, 0.01)  #正太分布初始化
                nn.init.constant_(m.bias, 0)        #初始化偏重为0


alexnet = AlexNet(num_classes=2)
summary(alexnet)