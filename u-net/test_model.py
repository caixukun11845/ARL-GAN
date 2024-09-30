import torch
from unet import U_NET_1
from torch import nn, optim, device
from train_model import train
from load_dataset import test_loader


U_net = U_NET_1()
def test(model, dataloader):
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.sigmoid(outputs)  # 使用sigmoid函数将输出映射到[0, 1]
            predicted = (predicted > 0.5).float()


test_result = test(U_net, test_loader)