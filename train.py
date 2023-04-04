import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os
from model import AlexNet
from train_tool import TrainTool


#使用GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#数据预处理
transform = {
    "train": transforms.Compose([transforms.Resize((224, 224)),   #随机裁剪，再缩放成224*224
                                 transforms.RandomHorizontalFlip(p=0.5), #水平方向随机翻转，概率为0.5
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]),
    "test": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])}
#导入加载数据集
#获取图像数据集的路径
# data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
data_root = os.getcwd()
image_path = data_root + "/data_road/"
print(image_path)
#导入训练集并进行处理
train_dataset = datasets.ImageFolder(root=image_path + "/train_road",
                                     transform=transform["train"])
train_num = len(train_dataset)
#加载训练集
train_loader = torch.utils.data.DataLoader(train_dataset,  #导入的训练集
                                           batch_size=32,  #每批训练样本个数
                                           shuffle=True,   #打乱训练集
                                           num_workers=0)  #使用线程数

#导入测试集并进行处理
test_dataset = datasets.ImageFolder(root=image_path + "/test_road",
                                     transform=transform["test"])
test_num = len(test_dataset)
#加载测试集
test_loader = torch.utils.data.DataLoader(test_dataset,  #导入的测试集
                                           batch_size=32,  #每批测试样本个数
                                           shuffle=True,   #打乱测试集
                                           num_workers=0)  #使用线程数

#定义超参数
alexnet = AlexNet(num_classes=2).to(device)   #定义网络模型
loss_function = nn.CrossEntropyLoss()  #定义损失函数为交叉熵
optimizer = optim.Adam(alexnet.parameters(), lr=0.0002)  #定义优化器定义参数学习率


#正式训练
train_acc = []
train_loss = []
test_acc = []
test_loss = []

epoch = 0

#for epoch in range(epochs):
while True:
    epoch = epoch + 1;
    alexnet.train()

    epoch_train_acc, epoch_train_loss = TrainTool.train(train_loader, alexnet, optimizer, loss_function, device)

    alexnet.eval()
    epoch_test_acc, epoch_test_loss = TrainTool.test(test_loader,alexnet, loss_function,device)

    # train_acc.append(epoch_train_loss)
    # train_loss.append(epoch_train_loss)
    # test_acc.append(epoch_test_acc)
    # test_loss.append(epoch_test_loss)
    if epoch_train_acc < 0.99 and epoch_test_acc < 0.99:
       template = ('Epoch:{:2d}, train_acc:{:.1f}%, train_loss:{:.2f}, test_acc:{:.1f}%, test_loss:{:.2f}')
       print(template.format(epoch, epoch_train_acc * 100, epoch_train_loss, epoch_test_acc * 100, epoch_test_loss))
       continue
    else:
       torch.save(alexnet.state_dict(),'./model/alexnet_params.pth')
       print('Done')
       break
