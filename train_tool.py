import torch
import matplotlib.pyplot as plt

class TrainTool:
      def image_show(images):
          print(images.shape)
          images = images.numpy()   #将图片有tensor转换成array
          print(images.shape)
          images = images.transpose((1, 2, 0))   # 将【3，224，256】-->【224,256,3】
          # std = [0.5, 0.5, 0.5]
          # mean = [0.5, 0.5, 0.5]
          # images = images * std + mean
          print(images.shape)
          plt.imshow(images)
          plt.show()


      def train(train_loader, model, optimizer, loss_function, device):
          train_size = len(train_loader.dataset)  # 训练集的大小
          num_batches = len(train_loader)  # 批次数目

          train_acc, train_loss = 0, 0  # 初始化正确率和损失
          # 获取图片及标签
          for images, labels in train_loader:
              images, labels = images.to(device), labels.to(device)
              # 计算预测误差
              pred_labels = model(images)
              loss = loss_function(pred_labels, labels)
              # 反向传播
              optimizer.zero_grad()  # grad属性归零
              loss.backward()  # 反向传播
              optimizer.step()  # 每一步自动更新
              # 记录acc和loss
              train_acc += (pred_labels.argmax(1) == labels).type(torch.float).sum().item()
              train_loss += loss.item()


          train_acc /= train_size
          train_loss /= num_batches

          return train_acc, train_loss



      def test(test_loader, model, loss_function, device):
          test_size = len(test_loader.dataset)  # 测试集的大小，一共10000张图片
          num_batches = len(test_loader)  # 批次数目，313（10000/32=312.5，向上取整）
          test_loss, test_acc = 0, 0

          # 当不进行训练时，停止梯度更新，节省计算内存消耗
          with torch.no_grad():
              for images, labels in test_loader:
                  images, labels = images.to(device), labels.to(device)

                  # 计算loss
                  pred_labels = model(images)
                  loss = loss_function(pred_labels, labels)

                  test_acc += (pred_labels.argmax(1) == labels).type(torch.float).sum().item()
                  test_loss += loss.item()

          test_acc /= test_size
          test_loss /= num_batches

          return test_acc, test_loss

