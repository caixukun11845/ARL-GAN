from unet import U_NET_1
from torch import nn, optim
from load_dataset import train_dataset_ct, train_loader_ct

U_net = U_NET_1()
criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失函数
optimizer = optim.Adam(U_net.parameters(), lr=0.001)
num_epochs = 30

def train(model, dataloader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        for images, labels in train_loader_ct:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # 清空梯度
            outputs = U_net(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")
