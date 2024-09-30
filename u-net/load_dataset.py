from torch.utils.data import Dataset, DataLoader, random_split
import torch
from PIL import Image
from torch.utils.tensorboard.summary import image
from torchvision import transforms
import os
import h5py


class Cmf_Dateset(Dataset):
    def __init__(self, h5_file):
        self.file = h5py.File(h5_file, 'r')
        self.images = self.file['images'][:]
        self.labels = self.file['masks'][:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        label = torch.tensor(self.labels[idx])
        return image, label


h5_file_adress_ct = "D:/python工程/dataset/cmf/train/unpaired_ct.h5"
h5_file_adress_mri = "D:/python工程/dataset/cmf/train/unpaired_mri.h5"
h5_file_adress_test = "D:/python工程/dataset/cmf/test/paired_mri_ct.h5"
cmf_dataset_ct = Cmf_Dateset(h5_file_adress_ct)
cmf_dataset_mri = Cmf_Dateset(h5_file_adress_mri)
cmf_dataset_test = Cmf_Dateset(h5_file_adress_test)

# # 定义图像和标签的变换
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # 调整图片大小为256x256
#     transforms.ToTensor()  # 转换为张量
# ])

# 分割ct数据集
train_size_ct = int(0.8 * len(cmf_dataset_ct))
val_size_ct = len(cmf_dataset_ct) - train_size_ct
train_dataset_ct, val_dataset_ct = random_split(cmf_dataset_ct, [train_size_ct, val_size_ct])

# 分割mri数据集
train_size_mri = int(0.8 * len(cmf_dataset_mri))
val_size_mri = len(cmf_dataset_mri) - train_size_mri
train_dataset_mri, val_dataset_mri = random_split(cmf_dataset_mri, [train_size_mri, val_size_mri])


# 加载ct数据集
train_loader_ct = DataLoader(train_dataset_ct, batch_size=16, shuffle=True)
val_loader_ct = DataLoader(val_dataset_ct, batch_size=16, shuffle=False)

# 加载mri数据集
train_loader_mri = DataLoader(train_dataset_mri, batch_size=16, shuffle=True)
val_loader_mri = DataLoader(val_dataset_mri, batch_size=16, shuffle=True)

# 加载测试数据集
test_loader = DataLoader(cmf_dataset_test, batch_size=16, shuffle=True)