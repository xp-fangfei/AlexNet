import torch
import torchvision.transforms as transforms
from PIL import Image
from model import AlexNet

#数据预处理
transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
#导入要测试的图像
image = Image.open('./data_road/val/100.jpg')
image = transforms(image)
image = torch.unsqueeze(image, dim=0)

#实例化网络，加载训练好的模型参数
model = AlexNet(num_classes=2)
model.load_state_dict(torch.load('./model/alexnet_params.pth'))

#预测
classes = ('cross', 'no_cross')


with torch.no_grad():
    outputs = model(image)
    print(outputs)
    predict = torch.max(outputs, dim=1)[1].data.numpy()
    print(predict)
print(classes[int(predict)])

