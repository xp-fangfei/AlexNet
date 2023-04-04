################################################
#   模型转换由.pth到.onnx
################################################

# #转换模型
# import torch
# from model import AlexNet
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)
#
# #实例化网络，加载训练好的模型参数
# model = AlexNet(num_classes=2).to(device)
# model.load_state_dict(torch.load('./model/alexnet_params.pth'))
#
# x = torch.randn(1, 3, 224, 224).to(device)
#
#
# with torch.no_grad():
#     torch.onnx.export(
#         model,                  # 要转换的模型
#         x,                      # 模型的任意一组输入
#         'alexnet_params_new.onnx',        # 导出的 ONNX 文件名
#         opset_version=11,       # ONNX 算子集版本
#         input_names=['input'],  # 输入 Tensor 的名称（自己起名字）
#         output_names=['output'] # 输出 Tensor 的名称（自己起名字）
#     )


################################################
#   模型验证是否加载成功
################################################

# #验证模型是否转换成功
# import onnx
#
# # 读取 ONNX 模型
# onnx_model = onnx.load('alexnet_params_new.onnx')
#
# # 检查模型格式是否正确
# onnx.checker.check_model(onnx_model)
# print('无报错，onnx模型载入成功')
#
# #以可读形式打印计算图
# print(onnx.helper.printable_graph(onnx_model.graph))


################################################
#   应用转换好的模型进行验证
################################################

#应用转换完成的模型
import onnxruntime
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

#载入onnx模型，获取onnx runtime推理器
ort_session = onnxruntime.InferenceSession('alexnet_params_new.onnx')
#预处理测试机图片预处理：缩放、裁剪、归一化、Tensor
# 数据预处理
transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
#导入要测试的图像
image = Image.open('./data_road/val/403.jpg')
input_image = transforms(image)
input_tensor = input_image.unsqueeze(0).numpy()
# print(input_tensor.shape)

# onnx runtime 输入
ort_inputs = {'input': input_tensor}
# onnx runtime 输出
pred_logits = ort_session.run(['output'], ort_inputs)[0]
pred_logits = torch.tensor(pred_logits)
# print(pred_logits.shape)


classes = ('cross', 'no_cross')
predict = torch.max(pred_logits, dim=1)[1].data.numpy()
print(predict)