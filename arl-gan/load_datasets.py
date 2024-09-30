from torch.utils.data import Dataset,DataLoader
import torch
from PIL import Image
from torch.utils.tensorboard.summary import image
from torchvision import transforms
import os

class Cmf_Dateset(Dataset):
    def __init__(self, img_dir, label_dir, transform = None):
        self.img_dir = img_dir # 图像路径
        self.label_dir = label_dir # 标签路径
        self.transform = transform  # 图像变换（处理）
        self.image_files_list = os.listdir(img_dir)  # 获取图像文件名列表

    def __len__(self):
        return len(self.image_files_list) # 获取图像文件名列表长度

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files_list[idx])   # 获取每幅图像路径
        label_path = os.path.join(self.label_dir, self.image_files_list[idx])  #获取标签路径

        image = Image.open(img_path).convert("RGB")  # 将图像变成rgb图
        label = Image.open(label_path).convert("L")  # 将标签变成灰度图 0~255

        if self.transform:
            image = self.transform(image) #对图像和标签进行变换
            label = self.transform(label)

        return image, label


train_image_dir = "D:/python工程/dataset/cmf/train"
train_label_dir = "D:/python工程/dataset/cmf/train"

val_image_dir = "D:/python工程/dataset/cmf/test"
val_label_dir = "D:/python工程/dataset/cmf/test"


# 定义图像和标签的变换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图片大小为256x256
    transforms.ToTensor()  # 转换为张量
])

# 创建数据集和数据加载器
train_dataset = Cmf_Dateset(img_dir=train_image_dir, label_dir=train_label_dir, transform=transform)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)

val_dataset = Cmf_Dateset(img_dir=val_image_dir, label_dir=val_label_dir, transform=transform)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=True)