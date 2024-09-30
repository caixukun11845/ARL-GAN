from torch.utils.tensorboard.summary import image
from unet import U_NET_1
from torch import nn, optim
from train_model import train
from test_model import test
from load_dataset import train_loader_ct
import torch

train_rate = 0.001
num_epochs = 30

def main():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型、损失函数和优化器
    model = U_NET_1()  # 21类分割任务（Pascal VOC）
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_rate)


    # 训练模型
    train(model, train_loader_ct, criterion, optimizer, device, num_epochs)

    # 保存模型
    model_save_path = 'unet_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")